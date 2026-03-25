from fastapi import FastAPI
from loguru import logger
import asyncio
from datetime import datetime

from config import CCXT_SYMBOLS, TIMEFRAME
from data_fetcher import watch_ohlcv_forever, fetch_ohlcv
from model import load_model, save_model, train_step, predict, create_model
from trader import get_current_position, close_position, place_order_with_risk
from telegram_bot import init_telegram, send_message

app = FastAPI(title="Bybit NN Trader v3 — WebSocket Real-time")

# Глобальные модели и оптимизаторы по символам
models = {}
optimizers = {}

async def process_new_candle(df: pd.DataFrame, symbol: str):
    """Вызывается при получении новой свечи по WebSocket"""
    try:
        logger.info(f"Новая свеча для {symbol} | Close: {df['close'].iloc[-1]:.2f}")

        # Дообучение
        if symbol not in models:
            models[symbol] = load_model(symbol)
            optimizers[symbol] = torch.optim.AdamW(models[symbol].parameters(), lr=1e-4)

        model = models[symbol]
        optimizer = optimizers[symbol]

        train_step(model, optimizer, df.tail(800))
        save_model(model, symbol)

        # Предсказание
        pred_price = predict(model, df)
        current_price = df['close'].iloc[-1]
        signal = 1 if pred_price > current_price * 1.002 else -1

        current_pos = get_current_position(symbol)

        if signal == 1 and current_pos != 'long':
            await close_position(symbol)
            await place_order_with_risk(symbol, 'buy', model)
            await send_message(f"🚀 <b>LONG</b> {symbol} @ {current_price:.2f} | Pred: {pred_price:.2f}")
        elif signal == -1 and current_pos != 'short':
            await close_position(symbol)
            await place_order_with_risk(symbol, 'sell', model)
            await send_message(f"🔻 <b>SHORT</b> {symbol} @ {current_price:.2f} | Pred: {pred_price:.2f}")
        else:
            logger.info(f"{symbol} — сигнал без изменений")

    except Exception as e:
        logger.error(f"Ошибка обработки свечи {symbol}: {e}")

@app.on_event("startup")
async def startup():
    await init_telegram()
    await send_message("✅ Бот запущен с <b>WebSocket реал-тайм</b>")

    # Запускаем WebSocket для каждого символа
    for symbol in CCXT_SYMBOLS:
        asyncio.create_task(watch_ohlcv_forever(symbol, process_new_candle))
        logger.info(f"Запущен WebSocket таск для {symbol}")

    # Первичное обучение (на всякий случай)
    for symbol in CCXT_SYMBOLS:
        df = fetch_ohlcv(symbol, limit=2000)
        if symbol not in models:
            models[symbol] = load_model(symbol)
            optimizers[symbol] = torch.optim.AdamW(models[symbol].parameters(), lr=1e-4)
        for _ in range(5):
            train_step(models[symbol], optimizers[symbol], df)
        save_model(models[symbol], symbol)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)