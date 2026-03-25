import os
from dotenv import load_dotenv

load_dotenv()

# Bybit
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
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

CCXT_SYMBOLS = [f"{s}/USDT:USDT" for s in SYMBOLS]