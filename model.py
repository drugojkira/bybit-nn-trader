import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from loguru import logger


# === Feature Engineering: технические индикаторы ===

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands: upper, middle, lower, %B"""
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    pct_b = (series - lower) / (upper - lower + 1e-10)
    return upper, middle, lower, pct_b


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range"""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Отношение текущего объёма к среднему"""
    return volume / (volume.rolling(period).mean() + 1e-10)


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    Подготовка расширенного набора фичей:
    OHLCV (5) + RSI (1) + MACD (3) + Bollinger %B (1) + ATR (1) + Volume Ratio (1) = 12 фичей
    """
    features = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # Технические индикаторы
    features['rsi'] = compute_rsi(df['close'])
    macd_line, signal_line, histogram = compute_macd(df['close'])
    features['macd'] = macd_line
    features['macd_signal'] = signal_line
    features['macd_hist'] = histogram
    _, _, _, pct_b = compute_bollinger_bands(df['close'])
    features['bb_pct_b'] = pct_b
    features['atr'] = compute_atr(df)
    features['volume_ratio'] = compute_volume_ratio(df['volume'])

    # Убираем NaN строки (из-за rolling)
    features = features.dropna()
    return features.values


INPUT_SIZE = 12  # Количество фичей
LOOKBACK = 60


class LSTMModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=128, num_layers=3, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        return self.fc2(out)


def get_model_path(symbol: str):
    import os
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, f"model_{symbol.replace('/', '_')}.pth")


def create_model():
    return LSTMModel()


def normalize_data(data: np.ndarray) -> tuple:
    """
    Robust нормализация: (x - median) / IQR
    Возвращает нормализованные данные + параметры для обратной трансформации
    """
    median = np.median(data, axis=0)
    q75 = np.percentile(data, 75, axis=0)
    q25 = np.percentile(data, 25, axis=0)
    iqr = q75 - q25 + 1e-8
    normalized = (data - median) / iqr
    return normalized, median, iqr


def train_step(model, optimizer, df, lookback=LOOKBACK):
    """Один шаг обучения с валидационным сплитом"""
    if len(df) < lookback + 50:
        return 0.0, 0.0  # train_loss, val_loss

    data = prepare_features(df)
    if len(data) < lookback + 50:
        return 0.0, 0.0

    # Robust нормализация
    normalized, _, _ = normalize_data(data)

    # Создание последовательностей
    X = [normalized[i:i + lookback] for i in range(len(normalized) - lookback)]
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(normalized[lookback:, 3:4], dtype=torch.float32)  # close

    # Train/val split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Training
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = nn.MSELoss()(pred, y_train)
    loss.backward()
    # Gradient clipping для стабильности
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = nn.MSELoss()(val_pred, y_val).item()

    train_loss = loss.item()
    logger.info(
        f"[{df['timestamp'].iloc[-1]}] Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}"
    )
    return train_loss, val_loss


def predict(model, df, lookback=LOOKBACK):
    """Предсказание следующей цены закрытия"""
    model.eval()
    data = prepare_features(df)
    if len(data) < lookback:
        logger.warning(f"Недостаточно данных для предсказания: {len(data)} < {lookback}")
        return None

    normalized, median, iqr = normalize_data(data)
    X = torch.tensor([normalized[-lookback:]], dtype=torch.float32)

    with torch.no_grad():
        pred_normalized = model(X)

    # Обратная трансформация (для close — индекс 3)
    pred_price = float(pred_normalized[0][0]) * iqr[3] + median[3]
    return pred_price


def save_model(model, symbol: str):
    path = get_model_path(symbol)
    torch.save(model.state_dict(), path)
    logger.info(f"Модель сохранена: {path}")


def load_model(symbol: str):
    model = create_model()
    path = get_model_path(symbol)
    try:
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        logger.info(f"Модель загружена: {path}")
    except Exception:
        logger.info(f"Новая модель для {symbol}")
    return model
