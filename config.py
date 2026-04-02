import os
from dotenv import load_dotenv

load_dotenv()

# Binance API (название переменных приведено в соответствие с кодом)
# Поддерживаем оба варианта .env: BINANCE_API_KEY и BYBIT_API_KEY (обратная совместимость)
API_KEY = os.getenv("BINANCE_API_KEY") or os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET") or os.getenv("BYBIT_API_SECRET")

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # Основной чат (обратная совместимость)
# Несколько чатов через запятую: "123456,789012"
TELEGRAM_CHAT_IDS = [
    cid.strip() for cid in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if cid.strip()
]
# Если TELEGRAM_CHAT_IDS не задан, используем TELEGRAM_CHAT_ID
if not TELEGRAM_CHAT_IDS and TELEGRAM_CHAT_ID:
    TELEGRAM_CHAT_IDS = [TELEGRAM_CHAT_ID]

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
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))  # Умеренная L2 регуляризация

# Anti-overfitting
TRAIN_EVERY_N_CANDLES = int(os.getenv("TRAIN_EVERY_N_CANDLES", 6))  # Обучаем каждые N свечей (6 = каждые 30мин при 5m TF)
NOISE_STD = float(os.getenv("NOISE_STD", 0.005))  # Gaussian noise injection
LR_SCHEDULER_FACTOR = float(os.getenv("LR_SCHEDULER_FACTOR", 0.5))  # Множитель снижения LR
LR_SCHEDULER_PATIENCE = int(os.getenv("LR_SCHEDULER_PATIENCE", 10))  # Шагов без улучшения до снижения LR

# Мульти-позиции
MAX_POSITIONS_PER_SYMBOL = int(os.getenv("MAX_POSITIONS_PER_SYMBOL", 1))  # Макс позиций на 1 символ
MAX_TOTAL_POSITIONS = int(os.getenv("MAX_TOTAL_POSITIONS", 5))  # Макс позиций всего

# Мониторинг
REPORT_INTERVAL_CANDLES = int(os.getenv("REPORT_INTERVAL_CANDLES", 50))
AUTO_DASHBOARD_INTERVAL = int(os.getenv("AUTO_DASHBOARD_INTERVAL", 200))

# Логирование
LOG_FILE = os.getenv("LOG_FILE", "logs/trader.log")
LOG_ROTATION = os.getenv("LOG_ROTATION", "50 MB")
LOG_RETENTION = os.getenv("LOG_RETENTION", "7 days")

CCXT_SYMBOLS = [f"{s[:-4]}/{s[-4:]}" for s in SYMBOLS]  # BTCUSDT -> BTC/USDT


# =========================================================
# V6 Ensemble — новая архитектура
# =========================================================

# Общие параметры
V6_LOOKBACK = int(os.getenv("V6_LOOKBACK", 60))  # Окно для TFT/TCN (свечей)
V6_TARGET_HORIZON = int(os.getenv("V6_TARGET_HORIZON", 6))  # Горизонт предсказания (свечей)
V6_DIRECTION_THRESHOLD = float(os.getenv("V6_DIRECTION_THRESHOLD", 0.001))  # Порог для классификации направления

# Пути
V6_MODEL_PATH = os.getenv("V6_MODEL_PATH", "saved_models_v6")
V6_REGISTRY_PATH = os.getenv("V6_REGISTRY_PATH", "model_registry")

# Расписание переобучения
V6_RETRAIN_QUICK_HOURS = float(os.getenv("V6_RETRAIN_QUICK_HOURS", 6))  # Quick retrain каждые N часов
V6_RETRAIN_FULL_HOURS = float(os.getenv("V6_RETRAIN_FULL_HOURS", 24))  # Full retrain каждые N часов

# TFT (Temporal Fusion Transformer)
V6_TFT_CONFIG = {
    'input_size': int(os.getenv("V6_TFT_INPUT_SIZE", 50)),
    'd_model': int(os.getenv("V6_TFT_D_MODEL", 64)),
    'nhead': int(os.getenv("V6_TFT_NHEAD", 4)),
    'num_layers': int(os.getenv("V6_TFT_NUM_LAYERS", 3)),
    'dropout': float(os.getenv("V6_TFT_DROPOUT", 0.1)),
    'num_classes': 3,
    'quantiles': [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
    'lookback': V6_LOOKBACK,
}

# LightGBM
V6_LGBM_CONFIG = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'num_leaves': int(os.getenv("V6_LGBM_NUM_LEAVES", 63)),
    'learning_rate': float(os.getenv("V6_LGBM_LR", 0.05)),
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': int(os.getenv("V6_LGBM_N_ESTIMATORS", 500)),
    'early_stopping_rounds': 50,
    'verbose': -1,
    'random_state': 42,
}

# TCN (Temporal Convolutional Network)
V6_TCN_CONFIG = {
    'input_size': int(os.getenv("V6_TCN_INPUT_SIZE", 50)),
    'num_channels': [64, 64, 32],
    'kernel_size': int(os.getenv("V6_TCN_KERNEL_SIZE", 3)),
    'dropout': float(os.getenv("V6_TCN_DROPOUT", 0.2)),
    'num_classes': 3,
    'lookback': V6_LOOKBACK,
}

# Regime Detector (HMM)
V6_REGIME_CONFIG = {
    'n_regimes': int(os.getenv("V6_REGIMES", 4)),
    'lookback': int(os.getenv("V6_REGIME_LOOKBACK", 200)),
    'retrain_interval': int(os.getenv("V6_REGIME_RETRAIN_INTERVAL", 500)),
}

# Training Pipeline
V6_TRAIN_CONFIG = {
    'n_splits': int(os.getenv("V6_N_SPLITS", 5)),
    'val_ratio': float(os.getenv("V6_VAL_RATIO", 0.15)),
    'embargo_bars': int(os.getenv("V6_EMBARGO_BARS", 12)),
    'meta_window': int(os.getenv("V6_META_WINDOW", 100)),
    'meta_temperature': float(os.getenv("V6_META_TEMPERATURE", 2.0)),
}

# Decision Engine
V6_MIN_AGREEMENT = float(os.getenv("V6_MIN_AGREEMENT", 0.5))
V6_MIN_RISK_REWARD = float(os.getenv("V6_MIN_RISK_REWARD", 1.5))
V6_COUNTER_TREND_SCALE = float(os.getenv("V6_COUNTER_TREND_SCALE", 0.5))
