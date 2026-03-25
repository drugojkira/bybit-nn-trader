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
    LOG_RETENTION,
)
from data_fetcher import watch_ohlcv_forever, fetch_ohlcv, fetch_ohlcv_paginated, get_atr
from model import load_model, save_model, train_step, predict, create_model
import trader
from trader import get_current_position, close_position, place_order_with_risk
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

# Глобальные модели и оптимизаторы по символам
models: dict = {}
optimizers: dict = {}
ws_tasks: list[asyncio.Task] = []

# Метрики качества предсказаний
prediction_history: dict[str, list] = {}
# Счётчик свечей для периодических отчётов
candle_counter: dict[str, int] = {}


def track_prediction(symbol: str, predicted: float, actual: float, signal: int):
    """Сохраняем историю предсказаний для мониторинга качества"""
    if symbol not in prediction_history:
        prediction_history[symbol] = []
    prediction_history[symbol].append((predicted, actual))
    if len(prediction_history[symbol]) > 500:
        prediction_history[symbol] = prediction_history[symbol][-500:]

    # Записываем в монитор обучения
    monitor.record_prediction(symbol, predicted, actual, signal)


def get_prediction_accuracy(symbol: str) -> dict:
    """Оценка точности направления предсказаний"""
    history = prediction_history.get(symbol, [])
    if len(history) < 10:
        return {'accuracy': None, 'mae': None, 'count': len(history)}

    correct = 0
    errors = []
    for i in range(1, len(history)):
        prev_actual = history[i - 1][1]
        predicted = history[i][0]
        actual = history[i][1]

        pred_dir = predicted > prev_actual
        actual_dir = actual > prev_actual
        if pred_dir == actual_dir:
            correct += 1
        errors.append(abs(predicted - actual))

    accuracy = correct / (len(history) - 1) if len(history) > 1 else 0
    mae = sum(errors) / len(errors) if errors else 0

    return {
        'accuracy': accuracy,
        'mae': mae,
        'count': len(history),
    }


async def process_new_candle(df: pd.DataFrame, symbol: str):
    """Вызывается при получении новой свечи по WebSocket"""
    try:
        current_price = df['close'].iloc[-1]
        logger.info(f"Новая свеча для {symbol} | Close: {current_price:.2f} | Свечей: {len(df)}")

        # Инициализация модели
        if symbol not in models:
            models[symbol] = load_model(symbol)
            optimizers[symbol] = torch.optim.AdamW(
                models[symbol].parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )

        model = models[symbol]
        optimizer = optimizers[symbol]

        # Дообучение с mini-batch и early stopping
        train_loss, val_loss = train_step(
            model, optimizer, df.tail(1000),
            batch_size=BATCH_SIZE,
            max_epochs=ONLINE_TRAIN_EPOCHS,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
        )

        # Записываем метрики обучения в монитор
        monitor.record_training(
            symbol=symbol,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=LEARNING_RATE,
        )

        # Сохраняем модель с версионированием (только если лучше)
        save_model(model, symbol, val_loss=val_loss, train_loss=train_loss)

        # Проверка overfitting с автоматическим оповещением
        if train_loss > 0 and val_loss > train_loss * 3:
            logger.warning(
                f"⚠️ Overfitting для {symbol}: train={train_loss:.6f}, val={val_loss:.6f}"
            )
            await send_message(
                f"⚠️ <b>Overfitting</b> {symbol}\n"
                f"Train: {train_loss:.6f}\n"
                f"Val: {val_loss:.6f}\n"
                f"Ratio: {val_loss / (train_loss + 1e-10):.1f}x"
            )

        # Предсказание
        pred_price = predict(model, df)
        if pred_price is None:
            logger.warning(f"Не удалось получить предсказание для {symbol}")
            return

        # Адаптивный порог на основе ATR
        atr = get_atr(df)
        threshold = calculate_adaptive_threshold(atr, current_price)

        # Генерация сигнала
        price_change = (pred_price - current_price) / current_price
        if price_change > threshold:
            signal = 1  # LONG
        elif price_change < -threshold:
            signal = -1  # SHORT
        else:
            signal = 0

        # Трекинг предсказаний
        track_prediction(symbol, pred_price, current_price, signal)

        # Проверка лимита просадки
        if not check_drawdown_limit(symbol):
            await send_message(
                f"⛔ <b>СТОП</b> {symbol}: превышен лимит просадки. Торговля приостановлена."
            )
            return

        # Проверяем, не приостановлена ли торговля
        if is_trading_paused():
            logger.info(f"{symbol} — торговля приостановлена | signal={signal}")
            return

        current_pos = await get_current_position(symbol)

        if signal == 1 and current_pos != 'long':
            await close_position(symbol)
            await place_order_with_risk(symbol, 'buy', model, prediction=pred_price)
            await send_message(
                f"🚀 <b>LONG</b> {symbol} @ {current_price:.2f}\n"
                f"Pred: {pred_price:.2f} ({price_change:+.2%})\n"
                f"Порог: {threshold:.4f} | ATR: {atr:.2f}\n"
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f}"
            )
        elif signal == -1 and current_pos != 'short':
            await close_position(symbol)
            await place_order_with_risk(symbol, 'sell', model, prediction=pred_price)
            await send_message(
                f"🔻 <b>SHORT</b> {symbol} @ {current_price:.2f}\n"
                f"Pred: {pred_price:.2f} ({price_change:+.2%})\n"
                f"Порог: {threshold:.4f} | ATR: {atr:.2f}\n"
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f}"
            )
        else:
            logger.info(
                f"{symbol} — без сигнала | Pred: {pred_price:.2f} "
                f"({price_change:+.4f} vs порог ±{threshold:.4f})"
            )

        # Счётчик свечей
        candle_counter[symbol] = candle_counter.get(symbol, 0) + 1

        # Периодический отчёт
        if candle_counter[symbol] % REPORT_INTERVAL_CANDLES == 0:
            stats = journal.get_statistics(symbol)
            accuracy = get_prediction_accuracy(symbol)
            training_summary = monitor.get_training_summary(symbol)

            report = (
                f"📊 <b>Отчёт {symbol}</b> (свеча #{candle_counter[symbol]})\n\n"
                f"<b>Торговля:</b>\n"
                f"Сделок: {stats['total_trades']} | "
                f"WR: {stats['win_rate']:.1%} | "
                f"PnL: {stats['total_pnl']:+.2f}%\n"
                f"Max DD: {stats['max_drawdown']:.2f}%\n\n"
                f"<b>Модель:</b>\n"
                f"Train: {training_summary.get('last_train_loss', 0):.6f} | "
                f"Val: {training_summary.get('last_val_loss', 0):.6f}\n"
                f"Loss тренд: {training_summary.get('loss_trend', 'N/A')}\n"
            )
            if accuracy['accuracy'] is not None:
                report += f"Точность направления: {accuracy['accuracy']:.1%}\n"
            await send_message(report)

        # Автоматический дашборд (реже)
        if candle_counter[symbol] % AUTO_DASHBOARD_INTERVAL == 0:
            dashboard = monitor.generate_full_dashboard(symbol)
            if dashboard:
                await send_photo(dashboard, f"📊 Авто-дашборд {symbol} (свеча #{candle_counter[symbol]})")

    except Exception as e:
        logger.error(f"Ошибка обработки свечи {symbol}: {e}")
        import traceback
        logger.debug(traceback.format_exc())


@app.on_event("startup")
async def startup():
    # Устанавливаем зависимости Telegram-бота (избегаем циклических импортов)
    set_dependencies(
        trader_mod=trader,
        monitor_mod=monitor_module,
        model_mod=model_module,
        journal_mod=__import__('trade_journal'),
    )

    await init_telegram()
    await send_message(
        "✅ <b>NN Trader v5</b> запущен\n"
        "WebSocket реал-тайм + Attention LSTM\n"
        "Используйте /start для списка команд"
    )

    # Первичное обучение
    for symbol in CCXT_SYMBOLS:
        try:
            logger.info(f"Загрузка данных для первичного обучения {symbol}...")
            df = fetch_ohlcv_paginated(symbol, total_limit=10000)
            logger.info(f"Загружено {len(df)} свечей для {symbol}")

            if symbol not in models:
                models[symbol] = load_model(symbol)
                optimizers[symbol] = torch.optim.AdamW(
                    models[symbol].parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
                )

            best_val = float('inf')
            for epoch in range(INITIAL_TRAIN_EPOCHS):
                train_loss, val_loss = train_step(
                    models[symbol], optimizers[symbol], df,
                    batch_size=BATCH_SIZE,
                    max_epochs=1,
                    early_stopping_patience=EARLY_STOPPING_PATIENCE,
                )
                monitor.record_training(symbol, train_loss, val_loss, epoch=epoch)

                if val_loss < best_val:
                    best_val = val_loss

                logger.info(
                    f"Первичное обучение {symbol}: epoch {epoch + 1}/{INITIAL_TRAIN_EPOCHS}, "
                    f"train={train_loss:.6f}, val={val_loss:.6f}"
                )

            save_model(models[symbol], symbol, val_loss=best_val)
            monitor.save_history()

            await send_message(
                f"🧠 Первичное обучение <b>{symbol}</b> завершено\n"
                f"Epochs: {INITIAL_TRAIN_EPOCHS} | Best Val: {best_val:.6f}\n"
                f"Данные: {len(df)} свечей"
            )

        except Exception as e:
            logger.error(f"Ошибка первичного обучения {symbol}: {e}")
            await send_message(f"❌ Ошибка обучения {symbol}: {e}")

    # Запускаем WebSocket
    for symbol in CCXT_SYMBOLS:
        task = asyncio.create_task(watch_ohlcv_forever(symbol, process_new_candle))
        ws_tasks.append(task)
        logger.info(f"Запущен WebSocket таск для {symbol}")


@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown"""
    logger.info("Получен сигнал завершения...")

    for task in ws_tasks:
        task.cancel()
    await asyncio.gather(*ws_tasks, return_exceptions=True)

    # Сохраняем модели и метрики
    for symbol, model in models.items():
        save_model(model, symbol)
    monitor.save_history()

    await send_message("🛑 Бот остановлен (graceful shutdown)")
    await shutdown_telegram()
    logger.info("Shutdown завершён")


# === Эндпоинты для мониторинга ===

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "symbols": len(CCXT_SYMBOLS),
        "trading_paused": is_trading_paused(),
        "uptime": datetime.utcnow().isoformat(),
    }


@app.get("/stats")
async def stats():
    """Статистика по всем символам"""
    result = {}
    for symbol in CCXT_SYMBOLS:
        result[symbol] = {
            "journal": journal.get_statistics(symbol),
            "prediction": get_prediction_accuracy(symbol),
            "position": await get_current_position(symbol),
            "training": monitor.get_training_summary(symbol),
        }
    return result


@app.get("/stats/{symbol}")
async def stats_symbol(symbol: str):
    """Статистика по конкретному символу"""
    ccxt_symbol = f"{symbol}/USDT"
    return {
        "journal": journal.get_statistics(ccxt_symbol),
        "prediction": get_prediction_accuracy(ccxt_symbol),
        "position": await get_current_position(ccxt_symbol),
        "training": monitor.get_training_summary(ccxt_symbol),
    }


@app.get("/training/{symbol}")
async def training_stats(symbol: str):
    """Метрики обучения"""
    ccxt_symbol = f"{symbol}/USDT"
    return {
        "training": monitor.get_training_summary(ccxt_symbol),
        "predictions": monitor.get_prediction_summary(ccxt_symbol),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
