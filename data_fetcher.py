import ccxt
import ccxt.pro as ccxtpro
import pandas as pd
import asyncio
from loguru import logger
from config import API_KEY, API_SECRET, TESTNET, TIMEFRAME, CCXT_SYMBOLS

# REST клиент
exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})
if TESTNET:
    exchange.set_sandbox_mode(True)

def fetch_ohlcv(symbol: str, limit=1000):
    ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def get_atr(df: pd.DataFrame, period=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

# === WebSocket ===
async def watch_ohlcv_forever(symbol: str, callback):
    """Бесконечный WebSocket для одного символа"""
    ws_exchange = None
    while True:
        try:
            ws_exchange = ccxtpro.bybit({
                'apiKey': API_KEY,
                'secret': API_SECRET,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            if TESTNET:
                ws_exchange.set_sandbox_mode(True)

            logger.info(f"WebSocket запущен для {symbol}")
            while True:
                ohlcv = await ws_exchange.watch_ohlcv(symbol, TIMEFRAME)
                # Берём последние 200 свечей для стабильности
                df = pd.DataFrame(ohlcv[-200:], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                await callback(df, symbol)

        except Exception as e:
            logger.error(f"WebSocket ошибка для {symbol}: {e}")
            if ws_exchange:
                await ws_exchange.close()
            await asyncio.sleep(5)