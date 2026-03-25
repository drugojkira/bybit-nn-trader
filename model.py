import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import json
import shutil
from datetime import datetime
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
    Подготовка мультитаймфреймных фичей:
    5m:  OHLCV (5) + RSI (1) + MACD (3) + BB%B (1) + ATR (1) + VolRatio (1) = 12
    15m: RSI (1) + MACD (1) + MACDhist (1)                                   = 3
    1h:  RSI (1) + MACD (1) + MACDhist (1)                                   = 3
    Итого: 18 фичей
    """
    features = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # --- 5m индикаторы ---
    features['rsi'] = compute_rsi(df['close'])
    macd_line, signal_line, histogram = compute_macd(df['close'])
    features['macd'] = macd_line
    features['macd_signal'] = signal_line
    features['macd_hist'] = histogram
    _, _, _, pct_b = compute_bollinger_bands(df['close'])
    features['bb_pct_b'] = pct_b
    features['atr'] = compute_atr(df)
    features['volume_ratio'] = compute_volume_ratio(df['volume'])

    # --- 15m индикаторы (ресемплинг из 5m) ---
    if 'timestamp' in df.columns:
        df_indexed = df.set_index('timestamp')
    else:
        df_indexed = df.copy()

    df_15m = df_indexed.resample('15min').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna()
    df_1h = df_indexed.resample('1h').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna()

    rsi_15m = compute_rsi(df_15m['close'])
    macd_15m, _, hist_15m = compute_macd(df_15m['close'])

    rsi_1h = compute_rsi(df_1h['close'])
    macd_1h, _, hist_1h = compute_macd(df_1h['close'])

    # Маппинг старших TF обратно на 5m (forward fill)
    features['rsi_15m'] = rsi_15m.reindex(df_indexed.index, method='ffill').values
    features['macd_15m'] = macd_15m.reindex(df_indexed.index, method='ffill').values
    features['hist_15m'] = hist_15m.reindex(df_indexed.index, method='ffill').values

    features['rsi_1h'] = rsi_1h.reindex(df_indexed.index, method='ffill').values
    features['macd_1h'] = macd_1h.reindex(df_indexed.index, method='ffill').values
    features['hist_1h'] = hist_1h.reindex(df_indexed.index, method='ffill').values

    features = features.dropna()
    return features.values


INPUT_SIZE = 18  # 12 (5m) + 3 (15m) + 3 (1h)
LOOKBACK = 60


class AttentionLayer(nn.Module):
    """Слой внимания поверх LSTM для улавливания значимых временных шагов"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attn_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_output * attn_weights, dim=1)  # (batch, hidden_size)
        return context, attn_weights.squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=64, num_layers=2,
                 output_size=1, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.attention = AttentionLayer(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        context, attn_weights = self.attention(lstm_out)
        out = self.dropout(context)
        out = self.relu(self.fc1(out))
        return self.fc2(out)


# === Версионирование моделей ===

MODELS_DIR = "models"
MAX_MODEL_VERSIONS = 10  # Максимум хранимых версий


def get_model_dir(symbol: str) -> str:
    """Директория для конкретного символа"""
    safe_name = symbol.replace('/', '_').replace(':', '_')
    model_dir = os.path.join(MODELS_DIR, safe_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def get_model_path(symbol: str) -> str:
    """Путь к текущей (лучшей) модели"""
    return os.path.join(get_model_dir(symbol), "model_best.pth")


def get_model_meta_path(symbol: str) -> str:
    """Путь к метаданным модели"""
    return os.path.join(get_model_dir(symbol), "meta.json")


def save_model(model, symbol: str, val_loss: float = None, train_loss: float = None,
               epoch: int = None):
    """
    Сохраняет модель с версионированием.
    Каждая версия нумеруется, хранится до MAX_MODEL_VERSIONS последних.
    """
    model_dir = get_model_dir(symbol)
    best_path = os.path.join(model_dir, "model_best.pth")

    # Проверяем, лучше ли новая модель
    meta_path = get_model_meta_path(symbol)
    should_save_best = True
    if val_loss is not None and os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        if meta.get('best_val_loss') is not None and val_loss >= meta['best_val_loss']:
            should_save_best = False

    # Сохраняем версию
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    version_path = os.path.join(model_dir, f"model_v_{timestamp}.pth")
    torch.save(model.state_dict(), version_path)

    # Обновляем best если нужно
    if should_save_best:
        torch.save(model.state_dict(), best_path)
        logger.info(f"Модель {symbol} сохранена как best (val_loss={val_loss:.6f})" if val_loss else
                    f"Модель {symbol} сохранена как best")

    # Обновляем метаданные
    meta = {
        'symbol': symbol,
        'last_update': timestamp,
        'best_val_loss': val_loss if (should_save_best and val_loss is not None) else (
            json.load(open(meta_path)).get('best_val_loss') if os.path.exists(meta_path) else val_loss
        ),
        'last_train_loss': train_loss,
        'last_val_loss': val_loss,
        'epoch': epoch,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Очищаем старые версии
    _cleanup_old_versions(model_dir)


def _cleanup_old_versions(model_dir: str):
    """Удаляет старые версии, оставляя MAX_MODEL_VERSIONS последних"""
    versions = sorted([
        f for f in os.listdir(model_dir)
        if f.startswith("model_v_") and f.endswith(".pth")
    ])
    while len(versions) > MAX_MODEL_VERSIONS:
        old = versions.pop(0)
        os.remove(os.path.join(model_dir, old))
        logger.debug(f"Удалена старая версия: {old}")


def rollback_model(symbol: str, steps_back: int = 1) -> bool:
    """Откатывает модель на N версий назад"""
    model_dir = get_model_dir(symbol)
    versions = sorted([
        f for f in os.listdir(model_dir)
        if f.startswith("model_v_") and f.endswith(".pth")
    ])

    if len(versions) < steps_back + 1:
        logger.warning(f"Недостаточно версий для отката {symbol}: {len(versions)} < {steps_back + 1}")
        return False

    target_version = versions[-(steps_back + 1)]
    target_path = os.path.join(model_dir, target_version)
    best_path = os.path.join(model_dir, "model_best.pth")

    shutil.copy2(target_path, best_path)
    logger.info(f"Модель {symbol} откачена к версии {target_version}")
    return True


def list_model_versions(symbol: str) -> list[dict]:
    """Список всех версий модели"""
    model_dir = get_model_dir(symbol)
    if not os.path.exists(model_dir):
        return []
    versions = sorted([
        f for f in os.listdir(model_dir)
        if f.startswith("model_v_") and f.endswith(".pth")
    ])
    result = []
    for v in versions:
        path = os.path.join(model_dir, v)
        result.append({
            'filename': v,
            'timestamp': v.replace('model_v_', '').replace('.pth', ''),
            'size_kb': os.path.getsize(path) / 1024,
        })
    return result


def create_model():
    return LSTMModel()


# === Нормализация (без data leakage) ===

def normalize_data(data: np.ndarray) -> tuple:
    """
    Robust нормализация: (x - median) / IQR
    Вычисляет параметры по переданным данным.
    """
    median = np.median(data, axis=0)
    q75 = np.percentile(data, 75, axis=0)
    q25 = np.percentile(data, 25, axis=0)
    iqr = q75 - q25 + 1e-8
    normalized = (data - median) / iqr
    return normalized, median, iqr


def apply_normalization(data: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    """Применяет уже рассчитанные параметры нормализации (для val/test)"""
    return (data - median) / iqr


# === Обучение с mini-batch и early stopping ===

class EarlyStopping:
    """Early stopping для предотвращения overfitting"""

    def __init__(self, patience: int = 5, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def check(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


def train_step(model, optimizer, df, lookback=LOOKBACK, batch_size=64,
               max_epochs=1, early_stopping_patience=5):
    """
    Обучение с mini-batch, правильным train/val split и early stopping.
    Нормализация вычисляется ТОЛЬКО по train данным (без data leakage).
    """
    if len(df) < lookback + 50:
        return 0.0, 0.0

    data = prepare_features(df)
    if len(data) < lookback + 50:
        return 0.0, 0.0

    # === Train/Val split ПЕРЕД нормализацией (fix data leakage) ===
    split_idx = int(len(data) * 0.7)
    train_data_raw = data[:split_idx]
    val_data_raw = data[split_idx:]

    # Нормализация ТОЛЬКО по train данным
    train_normalized, median, iqr = normalize_data(train_data_raw)
    val_normalized = apply_normalization(val_data_raw, median, iqr)

    # Создание последовательностей для train
    X_train_list, y_train_list = [], []
    for i in range(len(train_normalized) - lookback):
        X_train_list.append(train_normalized[i:i + lookback])
        y_train_list.append(train_normalized[i + lookback, 3:4])  # close

    if len(X_train_list) < 10:
        return 0.0, 0.0

    X_train = torch.tensor(np.array(X_train_list), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train_list), dtype=torch.float32)

    # Создание последовательностей для val
    # Для val используем последние lookback точек из train как контекст
    full_val = np.concatenate([train_normalized[-lookback:], val_normalized], axis=0)
    X_val_list, y_val_list = [], []
    for i in range(len(full_val) - lookback):
        X_val_list.append(full_val[i:i + lookback])
        y_val_list.append(full_val[i + lookback, 3:4])

    if len(X_val_list) < 5:
        return 0.0, 0.0

    X_val = torch.tensor(np.array(X_val_list), dtype=torch.float32)
    y_val = torch.tensor(np.array(y_val_list), dtype=torch.float32)

    # Mini-batch DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.MSELoss()
    es = EarlyStopping(patience=early_stopping_patience)

    final_train_loss = 0.0
    final_val_loss = 0.0

    for epoch in range(max_epochs):
        # Training
        model.train()
        epoch_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        final_train_loss = sum(epoch_losses) / len(epoch_losses)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            final_val_loss = loss_fn(val_pred, y_val).item()

        # Early stopping check
        if es.check(final_val_loss):
            logger.info(f"Early stopping на epoch {epoch + 1}/{max_epochs}")
            break

    if final_train_loss > 0 and final_val_loss > final_train_loss * 3:
        logger.warning(
            f"⚠️ Возможный overfitting: train={final_train_loss:.6f}, val={final_val_loss:.6f}"
        )

    return final_train_loss, final_val_loss


def predict(model, df, lookback=LOOKBACK):
    """Предсказание следующей цены закрытия"""
    model.eval()
    data = prepare_features(df)
    if len(data) < lookback:
        logger.warning(f"Недостаточно данных для предсказания: {len(data)} < {lookback}")
        return None

    # Нормализация — вычисляем по всем доступным данным (это predict, не train)
    normalized, median, iqr = normalize_data(data)
    X = torch.tensor([normalized[-lookback:]], dtype=torch.float32)

    with torch.no_grad():
        pred_normalized = model(X)

    # Обратная трансформация (для close — индекс 3)
    pred_price = float(pred_normalized[0][0]) * iqr[3] + median[3]
    return pred_price


def load_model(symbol: str):
    model = create_model()
    path = get_model_path(symbol)
    try:
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        logger.info(f"Модель загружена: {path}")
    except Exception:
        logger.info(f"Новая модель для {symbol}")
    return model
