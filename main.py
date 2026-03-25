import asyncio
import os
import pandas as pd
import torch
from fastapi import FastAPI
from loguru import logger
from datetime import datetime

from config import (
    CCXT_SYMBOLS, TIMEFRAME, INITIAL_TRAIN_EPOCHS, ONLINE_TRAIN_EPOCHS,
    BATCH_SIZE, EARLY_STOPPING_PATIENCE, LEARNING_RATE, WEIGHT_DECAY,
    REPORT_INTERVAL_CANDLES, AUTO_DASHBOARD_INTERVAL, LOG_FILE, LOG_ROTATION,
    LOG_RETENTION, TRAIN_EVERY_N_CANDLES, MAX_POSITIONS_PER_SYMBOL,
    MAX_TOTAL_POSITIONS, NOISE_STD, LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE,
)
from data_fetcher import watch_ohlcv_forever, fetch_ohlcv, fetch_ohlcv_paginated, get_atr
from model import load_model, save_model, train_step, predict, create_model
import trader
from trader import get_current_position, close_position, place_order_with_risk, get_all_positions
from risk_manager import calculate_adaptive_threshold, check_drawdown_limit
from trade_journal import journal
import telegram_bot
from telegram_bot import (
    init_telegram, shutdown_telegram, send_message, send_photo,
    is_trading_paused, set_dependencies,
)
from training_monitor import monitor
import model as model_module
import training_monitor as monitor_module

# Настройка логирования в файл
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logger.add(
    LOG_FILE,
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level="INFO",
)

app = FastAPI(title="NN Trader v5 — Enhanced")

# Глобальные модели, оптимизаторы и scheduler'ы
models: dict = {}
optimizers: dict = {}
schedulers: dict = {}
ws_tasks: list[asyncio.Task] = []

# Метрики качества предсказаний
prediction_history: dict[str, list] = {}
# Счётчик свечей для throttle обучения
candle_counter: dict[str, int] = {}


def _init_model_stack(symbol: str):
    """Инициализация модели + optimizer + scheduler для символа"""
    if symbol not in models:
        models[symbol] = load_model(symbol)
        optimizers[symbol] = torch.optim.AdamW(
            models[symbol].parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        # ReduceLROnPlateau — снижает LR когда val_loss перестаёт уменьшаться
        schedulers[symbol] = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizers[symbol],
            mode='min',
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            min_lr=1e-6,
            verbose=False,
        )


def track_prediction(symbol: str, predicted: float, actual: float, signal: int):
    if symbol not in prediction_history:
        prediction_history[symbol] = []
    prediction_history[symbol].append((predicted, actual))
    if len(prediction_history[symbol]) > 500:
        prediction_history[symbol] = prediction_history[symbol][-500:]
    monitor.record_prediction(symbol, predicted, actual, signal)


def get_prediction_accuracy(symbol: str) -> dict:
    history = prediction_history.get(symbol, [])
    if len(history) < 10:
        return {'accuracy': None, 'mae': None, 'count': len(history)}

    correct = 0
    errors = []
    for i in range(1, len(history)):
        prev_actual = history[i - 1][1]
        predicted = history[i][0]
        actual = history[i][1]
        if (predicted > prev_actual) == (actual > prev_actual):
            correct += 1
        errors.append(abs(predicted - actual))

    return {
        'accuracy': correct / (len(history) - 1) if len(history) > 1 else 0,
        'mae': sum(errors) / len(errors) if errors else 0,
        'count': len(history),
    }


async def process_new_candle(df: pd.DataFrame, symbol: str):
    """Вызывается при получении новой свечи по WebSocket"""
    try:
        current_price = df['close'].iloc[-1]
        candle_counter[symbol] = candle_counter.get(symbol, 0) + 1
        candle_num = candle_counter[symbol]

        logger.info(
            f"Свеча #{candle_num} для {symbol} | Close: {current_price:.2f} | "
            f"Буфер: {len(df)}"
        )

        _init_model_stack(symbol)
        model = models[symbol]
        optimizer = optimizers[symbol]
        scheduler = schedulers[symbol]

        # === THROTTLE ОБУЧЕНИЯ ===
        # Обучаем не на каждой свече, а каждые N свечей
        # Это ключевое изменение против overfitting
        train_loss, val_loss = 0.0, 0.0
        if candle_num % TRAIN_EVERY_N_CANDLES == 0:
            train_loss, val_loss = train_step(
                model, optimizer, scheduler, df.tail(2000),
                batch_size=BATCH_SIZE,
                max_epochs=ONLINE_TRAIN_EPOCHS,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                noise_std=NOISE_STD,
            )

            monitor.record_training(
                symbol=symbol,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=optimizer.param_groups[0]['lr'],
            )

            # Сохраняем модель с версионированием (только если лучше)
            save_model(model, symbol, val_loss=val_loss, train_loss=train_loss)

            # Логируем текущий LR
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"[{symbol}] Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {current_lr:.2e}"
            )

            # Предупреждение об overfitting
            if train_loss > 0 and val_loss > train_loss * 3:
                await send_message(
                    f"⚠️ <b>Overfitting</b> {symbol}\n"
                    f"Train: {train_loss:.6f}\n"
                    f"Val: {val_loss:.6f}\n"
                    f"LR: {current_lr:.2e}"
                )

        # === ПРЕДСКАЗАНИЕ (на каждой свече) ===
        pred_price = predict(model, df)
        if pred_price is None:
            return

        atr = get_atr(df)
        threshold = calculate_adaptive_threshold(atr, current_price)

        price_change = (pred_price - current_price) / current_price
        if price_change > threshold:
            signal = 1
        elif price_change < -threshold:
            signal = -1
        else:
            signal = 0

        track_prediction(symbol, pred_price, current_price, signal)

        # Проверка лимита просадки
        if not check_drawdown_limit(symbol):
            await send_message(
                f"⛔ <b>СТОП</b> {symbol}: превышен лимит просадки."
            )
            return

        if is_trading_paused():
            logger.info(f"{symbol} — торговля приостановлена | signal={signal}")
            return

        # === МУЛЬТИ-ПОЗИЦИИ ===
        current_pos = await get_current_position(symbol)
        all_positions = await get_all_positions()
        total_open = len(all_positions)

        # Проверяем лимиты позиций
        can_open_new = total_open < MAX_TOTAL_POSITIONS

        if signal == 1 and current_pos != 'long' and can_open_new:
            if current_pos is not None:
                await close_position(symbol)
            await place_order_with_risk(symbol, 'buy', model, prediction=pred_price)
            await send_message(
                f"🚀 <b>LONG</b> {symbol} @ {current_price:.2f}\n"
                f"Pred: {pred_price:.2f} ({price_change:+.2%})\n"
                f"ATR: {atr:.2f} | Позиций: {total_open + 1}/{MAX_TOTAL_POSITIONS}"
            )
        elif signal == -1 and current_pos != 'short' and can_open_new:
            if current_pos is not None:
                await close_position(symbol)
            await place_order_with_risk(symbol, 'sell', model, prediction=pred_price)
            await send_message(
                f"🔻 <b>SHORT</b> {symbol} @ {current_price:.2f}\n"
                f"Pred: {pred_price:.2f} ({price_change:+.2%})\n"
                f"ATR: {atr:.2f} | Позиций: {total_open + 1}/{MAX_TOTAL_POSITIONS}"
            )
        elif signal != 0 and not can_open_new:
            logger.info(
                f"{symbol} — сигнал {signal}, но лимит позиций "
                f"({total_open}/{MAX_TOTAL_POSITIONS})"
            )
        else:
            logger.info(
                f"{symbol} — без сигнала | Pred: {pred_price:.2f} "
                f"({price_change:+.4f} vs ±{threshold:.4f})"
            )

        # === Периодический отчёт ===
        if candle_num % REPORT_INTERVAL_CANDLES == 0:
            stats = journal.get_statistics(symbol)
            accuracy = get_prediction_accuracy(symbol)
            training_summary = monitor.get_training_summary(symbol)
            current_lr = optimizer.param_groups[0]['lr']

            report = (
                f"📊 <b>Отчёт {symbol}</b> (свеча #{candle_num})\n\n"
                f"<b>Торговля:</b>\n"
                f"Сделок: {stats['total_trades']} | "
                f"WR: {stats['win_rate']:.1%} | "
                f"PnL: {stats['total_pnl']:+.2f}%\n"
                f"Max DD: {stats['max_drawdown']:.2f}%\n\n"
                f"<b>Модель:</b>\n"
                f"Train: {training_summary.get('last_train_loss', 0):.6f} | "
                f"Val: {training_summary.get('last_val_loss', 0):.6f}\n"
                f"LR: {current_lr:.2e} | "
                f"Тренд: {training_summary.get('loss_trend', 'N/A')}\n"
            )
            if accuracy['accuracy'] is not None:
                report += f"Точность: {accuracy['accuracy']:.1%}\n"
            await send_message(report)

        # Авто-дашборд
        if candle_num % AUTO_DASHBOARD_INTERVAL == 0:
            dashboard = monitor.generate_full_dashboard(symbol)
            if dashboard:
                await send_photo(dashboard, f"📊 Дашборд {symbol} #{candle_num}")

    except Exception as e:
        logger.error(f"Ошибка обработки свечи {symbol}: {e}")
        import traceback
        logger.debug(traceback.format_exc())


@app.on_event("startup")
async def startup():
    set_dependencies(
        trader_mod=trader,
        monitor_mod=monitor_module,
        model_mod=model_module,
        journal_mod=__import__('trade_journal'),
    )

    await init_telegram()
    await send_message(
        "✅ <b>NN Trader v5.1</b> запущен\n"
        "Anti-overfitting + Multi-positions\n"
        f"Обучение каждые {TRAIN_EVERY_N_CANDLES} свечей\n"
        f"Макс позиций: {MAX_TOTAL_POSITIONS}\n"
        "Используйте /start для команд"
    )

    # Первичное обучение
    for symbol in CCXT_SYMBOLS:
        try:
            logger.info(f"Загрузка данных для {symbol}...")
            df = fetch_ohlcv_paginated(symbol, total_limit=10000)
            logger.info(f"Загружено {len(df)} свечей для {symbol}")

            _init_model_stack(symbol)

            best_val = float('inf')
            for epoch in range(INITIAL_TRAIN_EPOCHS):
                train_loss, val_loss = train_step(
                    models[symbol], optimizers[symbol], schedulers[symbol], df,
                    batch_size=BATCH_SIZE,
                    max_epochs=1,
                    early_stopping_patience=EARLY_STOPPING_PATIENCE,
                    noise_std=NOISE_STD,
                )
                monitor.record_training(symbol, train_loss, val_loss, epoch=epoch)

                if val_loss < best_val:
                    best_val = val_loss

                current_lr = optimizers[symbol].param_groups[0]['lr']
                logger.info(
                    f"Обучение {symbol}: epoch {epoch + 1}/{INITIAL_TRAIN_EPOCHS}, "
                    f"train={train_loss:.6f}, val={val_loss:.6f}, lr={current_lr:.2e}"
                )

            save_model(models[symbol], symbol, val_loss=best_val)
            monitor.save_history()

            await send_message(
                f"🧠 Обучение <b>{symbol}</b> завершено\n"
                f"Epochs: {INITIAL_TRAIN_EPOCHS} | Best Val: {best_val:.6f}\n"
                f"Данные: {len(df)} свечей"
            )

        except Exception as e:
            logger.error(f"Ошибка обучения {symbol}: {e}")
            await send_message(f"❌ Ошибка обучения {symbol}: {e}")

    # Запускаем WebSocket
    for symbol in CCXT_SYMBOLS:
        task = asyncio.create_task(watch_ohlcv_forever(symbol, process_new_candle))
        ws_tasks.append(task)
        logger.info(f"WebSocket запущен для {symbol}")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutdown...")
    for task in ws_tasks:
        task.cancel()
    await asyncio.gather(*ws_tasks, return_exceptions=True)

    for symbol, model in models.items():
        save_model(model, symbol)
    monitor.save_history()

    await send_message("🛑 Бот остановлен")
    await shutdown_telegram()
    logger.info("Shutdown завершён")


# === Эндпоинты ===

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "symbols": len(CCXT_SYMBOLS),
        "trading_paused": is_trading_paused(),
    }


@app.get("/stats")
async def stats():
    result = {}
    for symbol in CCXT_SYMBOLS:
        result[symbol] = {
            "journal": journal.get_statistics(symbol),
            "prediction": get_prediction_accuracy(symbol),
            "position": await get_current_position(symbol),
            "training": monitor.get_training_summary(symbol),
            "lr": optimizers[symbol].param_groups[0]['lr'] if symbol in optimizers else None,
        }
    return result


@app.get("/stats/{symbol}")
async def stats_symbol(symbol: str):
    ccxt_symbol = f"{symbol}/USDT"
    return {
        "journal": journal.get_statistics(ccxt_symbol),
        "prediction": get_prediction_accuracy(ccxt_symbol),
        "position": await get_current_position(ccxt_symbol),
        "training": monitor.get_training_summary(ccxt_symbol),
    }


@app.get("/training/{symbol}")
async def training_stats(symbol: str):
    ccxt_symbol = f"{symbol}/USDT"
    return {
        "training": monitor.get_training_summary(ccxt_symbol),
        "predictions": monitor.get_prediction_summary(ccxt_symbol),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
