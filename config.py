import os
from dotenv import load_dotenv

load_dotenv()

# Binance
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
TESTNET = os.getenv("TESTNET", "true").lower() == "true"

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Торговля
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTCUSDT").split(",")]
TIMEFRAME = os.getenv("TIMEFRAME", "5m")
INTERVAL_MINUTES = int(os.getenv("INTERVAL_MINUTES", 15))

# Риск-менеджмент
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))
FRACTIONAL_KELLY = float(os.getenv("FRACTIONAL_KELLY", 0.5))
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", 0.10))
USE_TRAILING = os.getenv("USE_TRAILING", "true").lower() == "true"
TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", 0.5))
SL_ATR_MULTIPLIER = float(os.getenv("SL_ATR_MULTIPLIER", 2.0))
TP_ATR_MULTIPLIER = float(os.getenv("TP_ATR_MULTIPLIER", 4.0))

# Новые параметры
SIGNAL_THRESHOLD_MIN = float(os.getenv("SIGNAL_THRESHOLD_MIN", 0.001))  # мин. порог 0.1%
SIGNAL_THRESHOLD_MAX = float(os.getenv("SIGNAL_THRESHOLD_MAX", 0.01))   # макс. порог 1.0%
KELLY_LOOKBACK_TRADES = int(os.getenv("KELLY_LOOKBACK_TRADES", 100))    # сделок для Kelly

CCXT_SYMBOLS = [f"{s[:-4]}/{s[-4:]}" for s in SYMBOLS]  # BTCUSDT -> BTC/USDT
