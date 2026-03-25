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
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    pct_b = (series - lower) / (upper - lower + 1e-10)
    return upper, middle, lower, pct_b


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume / (volume.rolling(period).mean() + 1e-10)


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    Мультитаймфреймные фичи:
    5m:  OHLCV (5) + RSI (1) + MACD (3) + BB%B (1) + ATR (1) + VolRatio (1) = 12
    15m: RSI (1) + MACD (1) + MACDhist (1)                                   = 3
    1h:  RSI (1) + MACD (1) + MACDhist (1)                                   = 3
    Итого: 18 фичей
    """
    features = df[['open', 'high', 'low', 'close', 'volume']].copy()

    features['rsi'] = compute_rsi(df['close'])
    macd_line, signal_line, histogram = compute_macd(df['close'])
    features['macd'] = macd_line
    features['macd_signal'] = signal_line
    features['macd_hist'] = histogram
    _, _, _, pct_b = compute_bollinger_bands(df['close'])
    features['bb_pct_b'] = pct_b
    features['atr'] = compute_atr(df)
    features['volume_ratio'] = compute_volume_ratio(df['volume'])

    # --- 15m и 1h индикаторы (ресемплинг из 5m) ---
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

    features['rsi_15m'] = rsi_15m.reindex(df_indexed.index, method='ffill').values
    features['macd_15m'] = macd_15m.reindex(df_indexed.index, method='ffill').values
    features['hist_15m'] = hist_15m.reindex(df_indexed.index, method='ffill').values
    features['rsi_1h'] = rsi_1h.reindex(df_indexed.index, method='ffill').values
    features['macd_1h'] = macd_1h.reindex(df_indexed.index, method='ffill').values
    features['hist_1h'] = hist_1h.reindex(df_indexed.index, method='ffill').values

    features = features.dropna()
    return features.values


INPUT_SIZE = 18
LOOKBACK = 60


# === Модель с Attention и anti-overfitting ===

class AttentionLayer(nn.Module):
    """Слой внимания поверх LSTM"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output):
        attn_weights = self.attention(lstm_output)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_output * attn_weights, dim=1)
        return context, attn_weights.squeeze(-1)


class LSTMModel(nn.Module):
    """
    LSTM + Attention с anti-overfitting дизайном:
    - Dropout между слоями и перед FC
    - BatchNorm для стабилизации обучения
    - Более широкая сеть (hidden=128) для 18 фичей
    """

    def __init__(self, input_size=INPUT_SIZE, hidden_size=128, num_layers=2,
                 output_size=1, dropout=0.3):
        super().__init__()
        # Input BatchNorm — нормализует входные фичи
        self.input_bn = nn.BatchNorm1d(input_size)

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.attention = AttentionLayer(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # FC с BatchNorm
        self.fc1 = nn.Linear(hidden_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size, seq_len, features = x.shape

        # BatchNorm по фичам (reshape для BN1d)
        x_flat = x.reshape(-1, features)          # (batch*seq, features)
        x_flat = self.input_bn(x_flat)
        x = x_flat.reshape(batch_size, seq_len, features)

        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        out = self.dropout(context)
        out = self.relu(self.bn1(self.fc1(out)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.fc2(out)))
        return self.fc3(out)


# === Версионирование моделей ===

MODELS_DIR = "models"
MAX_MODEL_VERSIONS = 10


def get_model_dir(symbol: str) -> str:
    safe_name = symbol.replace('/', '_').replace(':', '_')
    model_dir = os.path.join(MODELS_DIR, safe_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def get_model_path(symbol: str) -> str:
    return os.path.join(get_model_dir(symbol), "model_best.pth")


def get_model_meta_path(symbol: str) -> str:
    return os.path.join(get_model_dir(symbol), "meta.json")


def save_model(model, symbol: str, val_loss: float = None, train_loss: float = None,
               epoch: int = None):
    model_dir = get_model_dir(symbol)
    best_path = os.path.join(model_dir, "model_best.pth")

    meta_path = get_model_meta_path(symbol)
    should_save_best = True
    if val_loss is not None and os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        if meta.get('best_val_loss') is not None and val_loss >= meta['best_val_loss']:
            should_save_best = False

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    version_path = os.path.join(model_dir, f"model_v_{timestamp}.pth")
    torch.save(model.state_dict(), version_path)

    if should_save_best:
        torch.save(model.state_dict(), best_path)
        logger.info(f"Модель {symbol} сохранена как best (val_loss={val_loss:.6f})" if val_loss else
                    f"Модель {symbol} сохранена как best")

    prev_best = None
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            prev_best = json.load(f).get('best_val_loss')

    meta = {
        'symbol': symbol,
        'last_update': timestamp,
        'best_val_loss': val_loss if (should_save_best and val_loss is not None) else prev_best,
        'last_train_loss': train_loss,
        'last_val_loss': val_loss,
        'epoch': epoch,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    _cleanup_old_versions(model_dir)


def _cleanup_old_versions(model_dir: str):
    versions = sorted([
        f for f in os.listdir(model_dir)
        if f.startswith("model_v_") and f.endswith(".pth")
    ])
    while len(versions) > MAX_MODEL_VERSIONS:
        old = versions.pop(0)
        os.remove(os.path.join(model_dir, old))


def rollback_model(symbol: str, steps_back: int = 1) -> bool:
    model_dir = get_model_dir(symbol)
    versions = sorted([
        f for f in os.listdir(model_dir)
        if f.startswith("model_v_") and f.endswith(".pth")
    ])
    if len(versions) < steps_back + 1:
        return False
    target = versions[-(steps_back + 1)]
    shutil.copy2(os.path.join(model_dir, target), os.path.join(model_dir, "model_best.pth"))
    logger.info(f"Модель {symbol} откачена к {target}")
    return True


def list_model_versions(symbol: str) -> list[dict]:
    model_dir = get_model_dir(symbol)
    if not os.path.exists(model_dir):
        return []
    versions = sorted([
        f for f in os.listdir(model_dir)
        if f.startswith("model_v_") and f.endswith(".pth")
    ])
    return [
        {'filename': v, 'timestamp': v.replace('model_v_', '').replace('.pth', ''),
         'size_kb': os.path.getsize(os.path.join(model_dir, v)) / 1024}
        for v in versions
    ]


def create_model():
    return LSTMModel()


# === Нормализация (без data leakage) ===

def normalize_data(data: np.ndarray) -> tuple:
    """Robust нормализация: (x - median) / IQR. Вычисляет параметры по переданным данным."""
    median = np.median(data, axis=0)
    q75 = np.percentile(data, 75, axis=0)
    q25 = np.percentile(data, 25, axis=0)
    iqr = q75 - q25 + 1e-8
    normalized = (data - median) / iqr
    return normalized, median, iqr


def apply_normalization(data: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    """Применяет уже рассчитанные параметры нормализации (для val/test)"""
    return (data - median) / iqr


# === Обучение с anti-overfitting стратегиями ===

class EarlyStopping:
    """Early stopping с восстановлением лучших весов"""

    def __init__(self, patience: int = 5, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        self.should_stop = False

    def check(self, val_loss: float, model: nn.Module = None) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            if model:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False

    def restore_best(self, model: nn.Module):
        """Восстанавливает лучшие веса после early stopping"""
        if self.best_weights:
            model.load_state_dict(self.best_weights)


def _add_noise(tensor: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
    """Gaussian noise injection для регуляризации входных данных"""
    if noise_std > 0:
        noise = torch.randn_like(tensor) * noise_std
        return tensor + noise
    return tensor


def train_step(model, optimizer, scheduler, df, lookback=LOOKBACK, batch_size=64,
               max_epochs=1, early_stopping_patience=5, noise_std=0.005):
    """
    Обучение с комплексным anti-overfitting:
    1. Нормализация ТОЛЬКО по train (без data leakage)
    2. Sequential DataLoader (без shuffle — сохраняем временну́ю структуру)
    3. Gaussian noise injection на входные данные
    4. Early stopping с восстановлением лучших весов
    5. Huber loss вместо MSE (робастнее к выбросам)
    6. LR scheduler step после каждого вызова
    """
    if len(df) < lookback + 50:
        return 0.0, 0.0

    data = prepare_features(df)
    if len(data) < lookback + 50:
        return 0.0, 0.0

    # === Train/Val split ПЕРЕД нормализацией ===
    split_idx = int(len(data) * 0.7)
    train_data_raw = data[:split_idx]
    val_data_raw = data[split_idx:]

    # Нормализация ТОЛЬКО по train
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
    full_val = np.concatenate([train_normalized[-lookback:], val_normalized], axis=0)
    X_val_list, y_val_list = [], []
    for i in range(len(full_val) - lookback):
        X_val_list.append(full_val[i:i + lookback])
        y_val_list.append(full_val[i + lookback, 3:4])

    if len(X_val_list) < 5:
        return 0.0, 0.0

    X_val = torch.tensor(np.array(X_val_list), dtype=torch.float32)
    y_val = torch.tensor(np.array(y_val_list), dtype=torch.float32)

    # Sequential DataLoader — НЕ shuffle для временных рядов
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Huber loss — робастнее к выбросам чем MSE
    loss_fn = nn.HuberLoss(delta=1.0)
    es = EarlyStopping(patience=early_stopping_patience)

    final_train_loss = 0.0
    final_val_loss = 0.0

    for epoch in range(max_epochs):
        # === Training ===
        model.train()
        epoch_losses = []
        for batch_X, batch_y in train_loader:
            # Noise injection — регуляризация входных данных
            batch_X_noisy = _add_noise(batch_X, noise_std=noise_std)

            optimizer.zero_grad()
            pred = model(batch_X_noisy)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        final_train_loss = sum(epoch_losses) / len(epoch_losses)

        # === Validation (без noise) ===
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            final_val_loss = loss_fn(val_pred, y_val).item()

        # Early stopping с сохранением лучших весов
        if es.check(final_val_loss, model):
            logger.info(f"Early stopping на epoch {epoch + 1}/{max_epochs}")
            es.restore_best(model)
            break

    # Шаг scheduler'а ПОСЛЕ всех эпох (снижаем LR постепенно)
    if scheduler is not None:
        scheduler.step(final_val_loss)

    # Восстанавливаем лучшие веса если не было early stop
    if not es.should_stop and es.best_weights:
        es.restore_best(model)

    if final_train_loss > 0 and final_val_loss > final_train_loss * 3:
        logger.warning(
            f"⚠️ Overfitting: train={final_train_loss:.6f}, val={final_val_loss:.6f}"
        )

    return final_train_loss, final_val_loss


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
