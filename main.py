import asyncio
import signal
import pandas as pd
import torch
from fastapi import FastAPI
from loguru import logger
from datetime import datetime

from config import CCXT_SYMBOLS, TIMEFRAME
from data_fetcher import watch_ohlcv_forever, fetch_ohlcv, get_atr
from model import load_model, save_model, train_step, predict, create_model
from trader import get_current_position, close_position, place_order_with_risk
from risk_manager import calculate_adaptive_threshold, check_drawdown_limit
from trade_journal import journal
from telegram_bot import init_telegram, send_message

app = FastAPI(title="Bybit NN Trader v4 — Enhanced")

# Глобальные модели и оптимизаторы по символам
models: dict = {}
optimizers: dict = {}
ws_tasks: list[asyncio.Task] = []

# Метрики качества предсказаний
prediction_history: dict[str, list] = {}  # symbol -> [(predicted, actual), ...]


def track_prediction(symbol: str, predicted: float, actual: float):
    """Сохраняем историю предсказаний для мониторинга качества"""
    if symbol not in prediction_history:
        prediction_history[symbol] = []
    prediction_history[symbol].append((predicted, actual))
    # Храним последние 500
    if len(prediction_history[symbol]) > 500:
        prediction_history[symbol] = prediction_history[symbol][-500:]


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

        # Направление: верно ли предсказали рост/падение
        pred_dir = predicted > prev_actual
        actual_dir = actual > prev_actual
        if pred_dir == actual_dir:
            correct += 1

        # MAE
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
        logger.info(f"Новая свеча для {symbol} | Close: {current_price:.2f}")

        # Инициализация модели
        if symbol not in models:
            models[symbol] = load_model(symbol)
            optimizers[symbol] = torch.optim.AdamW(
                models[symbol].parameters(), lr=1e-4, weight_decay=1e-5
            )

        model = models[symbol]
        optimizer = optimizers[symbol]

        # Дообучение с валидацией
        train_loss, val_loss = train_step(model, optimizer, df.tail(800))

        # Проверка overfitting: если val_loss >> train_loss — логируем предупреждение
        if train_loss > 0 and val_loss > train_loss * 3:
            logger.warning(
                f"⚠️ Возможный overfitting для {symbol}: "
                f"train={train_loss:.6f}, val={val_loss:.6f}"
            )

        save_model(model, symbol)

        # Предсказание
        pred_price = predict(model, df)
        if pred_price is None:
            logger.warning(f"Не удалось получить предсказание для {symbol}")
            return

        # Трекинг предсказаний
        track_prediction(symbol, pred_price, current_price)

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
            signal = 0  # Нет сигнала (в зоне неопределённости)

        # Проверка лимита просадки
        if not check_drawdown_limit(symbol):
            await send_message(
                f"⛔ <b>СТОП</b> {symbol}: превышен лимит просадки. Торговля приостановлена."
            )
            return

        current_pos = get_current_position(symbol)

        if signal == 1 and current_pos != 'long':
            await close_position(symbol)
            await place_order_with_risk(symbol, 'buy', model, prediction=pred_price)
            await send_message(
                f"🚀 <b>LONG</b> {symbol} @ {current_price:.2f}\n"
                f"Pred: {pred_price:.2f} ({price_change:+.2%})\n"
                f"Порог: {threshold:.4f} | ATR: {atr:.2f}"
            )
        elif signal == -1 and current_pos != 'short':
            await close_position(symbol)
            await place_order_with_risk(symbol, 'sell', model, prediction=pred_price)
            await send_message(
                f"🔻 <b>SHORT</b> {symbol} @ {current_price:.2f}\n"
                f"Pred: {pred_price:.2f} ({price_change:+.2%})\n"
                f"Порог: {threshold:.4f} | ATR: {atr:.2f}"
            )
        else:
            logger.info(
                f"{symbol} — без сигнала | Pred: {pred_price:.2f} "
                f"({price_change:+.4f} vs порог ±{threshold:.4f})"
            )

        # Периодический отчёт (каждые 50 свечей)
        hist = prediction_history.get(symbol, [])
        if len(hist) % 50 == 0 and len(hist) > 0:
            stats = journal.get_statistics(symbol)
            accuracy = get_prediction_accuracy(symbol)
            report = (
                f"📊 <b>Отчёт {symbol}</b>\n"
                f"Сделок: {stats['total_trades']} | "
                f"WR: {stats['win_rate']:.1%} | "
                f"PnL: {stats['total_pnl']:+.2f}%\n"
                f"Точность направления: {accuracy['accuracy']:.1%}" if accuracy['accuracy'] else ""
            )
            await send_message(report)

    except Exception as e:
        logger.error(f"Ошибка обработки свечи {symbol}: {e}")
        import traceback
        logger.debug(traceback.format_exc())


@app.on_event("startup")
async def startup():
    await init_telegram()
    await send_message("✅ Бот v4 запущен с <b>WebSocket реал-тайм</b>")

    # Первичное обучение
    for symbol in CCXT_SYMBOLS:
        try:
            df = fetch_ohlcv(symbol, limit=2000)
            if symbol not in models:
                models[symbol] = load_model(symbol)
                optimizers[symbol] = torch.optim.AdamW(
                    models[symbol].parameters(), lr=1e-4, weight_decay=1e-5
                )
            for epoch in range(5):
                train_loss, val_loss = train_step(models[symbol], optimizers[symbol], df)
                logger.info(f"Первичное обучение {symbol}: epoch {epoch + 1}/5, "
                            f"train={train_loss:.6f}, val={val_loss:.6f}")
            save_model(models[symbol], symbol)
        except Exception as e:
            logger.error(f"Ошибка первичного обучения {symbol}: {e}")

    # Запускаем WebSocket для каждого символа
    for symbol in CCXT_SYMBOLS:
        task = asyncio.create_task(watch_ohlcv_forever(symbol, process_new_candle))
        ws_tasks.append(task)
        logger.info(f"Запущен WebSocket таск для {symbol}")


@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown: закрываем WebSocket-соединения"""
    logger.info("Получен сигнал завершения, закрываем соединения...")
    for task in ws_tasks:
        task.cancel()
    await asyncio.gather(*ws_tasks, return_exceptions=True)

    # Сохраняем все модели
    for symbol, model in models.items():
        save_model(model, symbol)

    await send_message("🛑 Бот остановлен (graceful shutdown)")
    logger.info("Shutdown завершён")


# === Эндпоинты для мониторинга ===

@app.get("/health")
async def health():
    return {"status": "ok", "symbols": len(CCXT_SYMBOLS)}


@app.get("/stats")
async def stats():
    """Статистика по всем символам"""
    result = {}
    for symbol in CCXT_SYMBOLS:
        result[symbol] = {
            "journal": journal.get_statistics(symbol),
            "prediction": get_prediction_accuracy(symbol),
            "position": get_current_position(symbol),
        }
    return result


@app.get("/stats/{symbol}")
async def stats_symbol(symbol: str):
    """Статистика по конкретному символу"""
    ccxt_symbol = f"{symbol}/USDT:USDT"
    return {
        "journal": journal.get_statistics(ccxt_symbol),
        "prediction": get_prediction_accuracy(ccxt_symbol),
        "position": get_current_position(ccxt_symbol),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
