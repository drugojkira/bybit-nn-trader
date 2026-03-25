import os
from dotenv import load_dotenv

load_dotenv()

# Binance API (название переменных приведено в соответствие с кодом)
# Поддерживаем оба варианта .env: BINANCE_API_KEY и BYBIT_API_KEY (обратная совместимость)
API_KEY = os.getenv("BINANCE_API_KEY") or os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET") or os.getenv("BYBIT_API_SECRET")

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
MIN_TRADES_FOR_KELLY = int(os.getenv("MIN_TRADES_FOR_KELLY", 10))

# Сигналы
SIGNAL_THRESHOLD_MIN = float(os.getenv("SIGNAL_THRESHOLD_MIN", 0.001))
SIGNAL_THRESHOLD_MAX = float(os.getenv("SIGNAL_THRESHOLD_MAX", 0.01))
KELLY_LOOKBACK_TRADES = int(os.getenv("KELLY_LOOKBACK_TRADES", 100))

# Обучение
INITIAL_TRAIN_EPOCHS = int(os.getenv("INITIAL_TRAIN_EPOCHS", 10))
ONLINE_TRAIN_EPOCHS = int(os.getenv("ONLINE_TRAIN_EPOCHS", 3))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 5))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-4))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-3))

# Мониторинг
REPORT_INTERVAL_CANDLES = int(os.getenv("REPORT_INTERVAL_CANDLES", 50))
AUTO_DASHBOARD_INTERVAL = int(os.getenv("AUTO_DASHBOARD_INTERVAL", 200))

# Логирование
LOG_FILE = os.getenv("LOG_FILE", "logs/trader.log")
LOG_ROTATION = os.getenv("LOG_ROTATION", "50 MB")
LOG_RETENTION = os.getenv("LOG_RETENTION", "7 days")

CCXT_SYMBOLS = [f"{s[:-4]}/{s[-4:]}" for s in SYMBOLS]  # BTCUSDT -> BTC/USDT
