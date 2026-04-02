# План новой модели: NN Trader v6

## Почему текущая модель не работает

### 7 фундаментальных проблем

**1. Предсказание сырой цены — бессмысленная задача**
Модель предсказывает нормализованный close, потом денормализует обратно. Цена — нестационарный ряд (среднее и дисперсия постоянно меняются). LSTM не может выучить паттерн в нестационарном ряде — он каждый раз видит новые масштабы. Это причина #1 почему модель переобучается: она заучивает конкретные ценовые уровни вместо паттернов.

**2. Одношаговое точечное предсказание**
Предсказать цену через 5 минут с точностью, достаточной для прибыльной торговли — практически невозможно. Шум на 5m таймфрейме огромен. Одно число (pred_price) не несёт информации о уверенности модели. Нужно знать: «я уверен на 85% что цена пойдёт вверх на 0.3-0.7%» — а не «цена будет 84523.17».

**3. Запаздывающие индикаторы**
RSI, MACD, Bollinger Bands — это lagging indicators. Они описывают прошлое, не будущее. Модель получает 18 фичей, из которых 15 — производные от одного и того же ценового ряда. Это pseudo-features: они не добавляют новой информации, только пересказывают цену разными словами.

**4. Нет понимания рыночного режима**
Рынок кардинально меняет поведение: тренд → боковик → высокая волатильность → крах. Одна модель с одними параметрами не может одинаково хорошо работать во всех режимах. Стратегия «купи когда предсказание выше» убыточна в боковике, а «не торгуй в боковике» убыточна в тренде.

**5. Online learning разрушает модель**
Обучение на каждой новой свече (даже каждые 6 свечей) вызывает catastrophic forgetting: модель забывает выученные паттерны, постоянно подстраиваясь под последние данные. Это не адаптация — это хаос.

**6. Нет оценки уверенности**
Модель выдаёт одно число. Нет способа отличить «я уверен» от «я гадаю». Пороговое значение (threshold) на основе ATR — грубый хак. Модель должна сама знать, когда она не знает.

**7. Нет правильной валидации**
Один train/val split (70/30) — ненадёжная оценка. Модель может показать хороший val_loss на одном периоде и полностью провалиться на другом. Для финансовых данных нужен walk-forward validation с множеством окон.

---

## Новая архитектура: Ensemble с Regime Detection

### Общая схема

```
                    ┌─────────────────────┐
                    │   Data Pipeline     │
                    │   (50+ фичей)       │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  Regime Detector    │
                    │  (HMM, 4 состояния) │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼──────┐ ┌─────▼──────┐ ┌──────▼─────────┐
    │  TFT Model     │ │ LightGBM   │ │  TCN Model     │
    │  (Transformer) │ │ (Gradient  │ │  (Temporal CNN) │
    │  Multi-horizon │ │  Boosting) │ │  Fast inference │
    └─────────┬──────┘ └─────┬──────┘ └──────┬─────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Meta-Learner      │
                    │ (взвешенное голосов.)│
                    │ + Confidence Score   │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   Trade Decision    │
                    │   Engine            │
                    │   (режим + сигнал   │
                    │    + уверенность    │
                    │    + риск)          │
                    └─────────────────────┘
```

---

## Фаза 1: Data Pipeline (новый data_fetcher + feature_engine)

### 1.1 Новые целевые переменные (targets)

**Вместо raw price → log returns + classification**

```python
# Текущий подход (ПЛОХО):
target = normalized_close_price[t+1]

# Новый подход:
# 1) Log return (для регрессии)
log_return_1  = log(close[t+1] / close[t])      # 1 свеча вперёд
log_return_6  = log(close[t+6] / close[t])      # 30 мин вперёд
log_return_12 = log(close[t+12] / close[t])     # 1 час вперёд

# 2) Normalized return (для стабильности)
norm_return = log_return / rolling_std(log_returns, 20)

# 3) Direction class (для классификации)
# Класс определяется по превышению порога (ATR-based)
direction = {
    +1: norm_return > +0.5   # значимый рост
    -1: norm_return < -0.5   # значимое падение
     0: иначе                # шум / боковик
}
```

**Почему это лучше:**
- Log returns стационарны (среднее ≈ 0, дисперсия стабильна)
- Normalized returns масштабируют по волатильности — одна и та же «сила сигнала» в спокойный и бурный рынок
- 3-классовая классификация (up/down/flat) фильтрует шум — модель не торгует когда не уверена
- Мульти-горизонт (1/6/12 свечей) позволяет видеть краткосрочный и среднесрочный тренд

### 1.2 Feature Engineering: 4 категории, 50+ фичей

**Категория A: Price Action (15 фичей)**
```
- log_return (1, 3, 6, 12, 24 лагов)           — 5 фичей
- rolling_volatility (10, 20, 50 периодов)       — 3 фичи
- high_low_range / close                          — 1 фича
- close_position_in_range (где close в рамках H-L)— 1 фича
- upper_shadow_ratio, lower_shadow_ratio          — 2 фичи
- body_ratio (|open-close| / (high-low))          — 1 фича
- gap (open[t] vs close[t-1])                     — 1 фича
- trend_strength (ADX)                            — 1 фича
```

**Категория B: Volume & Microstructure (10 фичей)**
```
- volume_ratio (volume / SMA_volume_20)           — 1 фича
- volume_delta (buy_vol - sell_vol, если доступно) — 1 фича
- OBV (On-Balance Volume)                         — 1 фича
- VWAP_distance (цена vs VWAP)                    — 1 фича
- volume_trend (наклон линейной регрессии vol)    — 1 фича
- large_volume_flag (volume > 2 * SMA)            — 1 фича
- funding_rate (Binance futures)                   — 1 фича
- open_interest_change                             — 1 фича
- long_short_ratio                                 — 1 фича
- liquidation_volume (если доступно через API)     — 1 фича
```

**Категория C: Technical Patterns (15 фичей)**
```
- RSI (14)                                        — 1 фича
- RSI_divergence (цена вверх + RSI вниз и наобор.)— 1 фича
- MACD_histogram                                  — 1 фича
- MACD_cross_signal (1/0/-1)                      — 1 фича
- BB_%B (позиция в Bollinger Bands)               — 1 фича
- BB_width (нормализованная ширина полос)          — 1 фича
- ATR_normalized (ATR / close)                    — 1 фича
- Stochastic_K, Stochastic_D                     — 2 фичи
- CCI (Commodity Channel Index)                   — 1 фича
- Williams_%R                                     — 1 фича
- Ichimoku_cloud_distance                         — 1 фича
- EMA_cross (9/21 cross signal)                   — 1 фича
- SMA_50_200_distance (distance to 50/200 SMA)   — 2 фичи
```

**Категория D: Multi-Timeframe Context (10+ фичей)**
```
# Вместо хака с resampling — запрашиваем отдельно 15m и 1h свечи
15m: RSI, MACD_hist, ATR_norm, trend_direction    — 4 фичи
1h:  RSI, MACD_hist, ATR_norm, trend_direction    — 4 фичи
4h:  RSI, trend_direction                         — 2 фичи
```

### 1.3 Получение данных

```python
# Новые источники данных (через ccxt и Binance API):
# 1. OHLCV с нескольких таймфреймов — ccxt.fetch_ohlcv()
# 2. Funding rate — ccxt.fetch_funding_rate()
# 3. Open Interest — ccxt.fetch_open_interest_history()
# 4. Long/Short ratio — Binance API endpoint
# 5. Liquidations — Binance WebSocket stream

# Хранение: буфер на 10000 свечей (5m = ~35 дней)
# Пагинированная загрузка при старте
# WebSocket для real-time обновлений
```

### 1.4 Нормализация (без data leakage)

```python
# Expanding window normalization для online-режима:
class ExpandingNormalizer:
    """
    Нормализация по расширяющемуся окну.
    На каждом шаге статистики вычисляются ТОЛЬКО по прошлым данным.
    """
    def __init__(self, min_periods=200):
        self.min_periods = min_periods

    def fit_transform(self, data, split_idx):
        # train: вычисляем stats по [0:split_idx]
        # val: вычисляем stats по [0:split_idx] (те же!)
        # test: вычисляем stats по [0:split_idx+val_size]
        train_data = data[:split_idx]
        median = np.median(train_data, axis=0)
        iqr = np.percentile(train_data, 75, axis=0) - np.percentile(train_data, 25, axis=0) + 1e-8
        return (data - median) / iqr, median, iqr
```

---

## Фаза 2: Regime Detection (новый файл: regime_detector.py)

### 2.1 Hidden Markov Model (4 состояния)

```python
from hmmlearn import hmm

class RegimeDetector:
    """
    Определяет текущий рыночный режим:
    - TRENDING_UP:    устойчивый рост, низкая/средняя волатильность
    - TRENDING_DOWN:  устойчивое падение, растущая волатильность
    - RANGING:        боковик, низкая волатильность
    - VOLATILE:       хаос, высокая волатильность, резкие развороты
    """

    REGIMES = {0: 'trending_up', 1: 'trending_down', 2: 'ranging', 3: 'volatile'}

    def __init__(self, n_states=4, lookback=500):
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='full',
            n_iter=100,
            random_state=42
        )
        self.lookback = lookback
        self.fitted = False

    def _prepare_features(self, df):
        """Фичи для определения режима"""
        returns = np.log(df['close'] / df['close'].shift(1))
        volatility = returns.rolling(20).std()
        trend = returns.rolling(50).mean() / (volatility + 1e-10)
        volume_ratio = df['volume'] / df['volume'].rolling(20).mean()

        features = np.column_stack([
            returns.values,
            volatility.values,
            trend.values,
            volume_ratio.values
        ])
        return features[~np.isnan(features).any(axis=1)]

    def fit(self, df):
        features = self._prepare_features(df)
        self.model.fit(features[-self.lookback:])
        self.fitted = True

    def predict_regime(self, df) -> str:
        if not self.fitted:
            return 'unknown'
        features = self._prepare_features(df)
        state = self.model.predict(features[-1:])
        return self.REGIMES[state[0]]

    def get_regime_params(self, regime: str) -> dict:
        """Адаптивные параметры торговли по режиму"""
        params = {
            'trending_up': {
                'trade': True,
                'prefer_direction': 'long',
                'position_scale': 1.2,      # больше позиция в тренде
                'sl_multiplier': 2.5,       # широкий SL в тренде
                'tp_multiplier': 5.0,       # далёкий TP
                'min_confidence': 0.55,     # ниже порог входа
            },
            'trending_down': {
                'trade': True,
                'prefer_direction': 'short',
                'position_scale': 1.2,
                'sl_multiplier': 2.5,
                'tp_multiplier': 5.0,
                'min_confidence': 0.55,
            },
            'ranging': {
                'trade': True,
                'prefer_direction': None,    # обе стороны
                'position_scale': 0.7,       # меньше позиция
                'sl_multiplier': 1.5,        # узкий SL
                'tp_multiplier': 2.0,        # близкий TP (mean reversion)
                'min_confidence': 0.70,      # высокий порог — торгуем редко
            },
            'volatile': {
                'trade': False,              # НЕ ТОРГУЕМ в хаосе
                'prefer_direction': None,
                'position_scale': 0.0,
                'sl_multiplier': 0,
                'tp_multiplier': 0,
                'min_confidence': 1.0,       # по факту — не торгуем
            },
        }
        return params.get(regime, params['ranging'])
```

### 2.2 Переобучение Regime Detector

```
- Переобучается каждые 24 часа (не на каждой свече!)
- Использует последние 500-1000 свечей
- Проверка: если режим меняется слишком часто (>5 раз в час) —
  фиксируем RANGING как fallback
```

---

## Фаза 3: Core Models (новый model.py)

### 3.1 Модель 1: Temporal Fusion Transformer (TFT)

**Почему TFT, а не LSTM:**
- Встроенный механизм Variable Selection — модель сама выбирает важные фичи
- Multi-horizon prediction — одна модель предсказывает на 1, 6, 12 шагов вперёд
- Interpretable Attention — видно какие временны́е шаги важны
- Gated Residual Network — устойчив к overfitting
- Quantile output — вместо одного числа даёт распределение (10%, 50%, 90%)

```python
# Используем pytorch-forecasting (проверенная реализация Google TFT)
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Конфигурация TFT
tft_config = {
    'hidden_size': 64,               # размер скрытого состояния
    'attention_head_size': 4,         # голов внимания
    'dropout': 0.2,                   # dropout
    'hidden_continuous_size': 32,     # для непрерывных переменных
    'output_size': 7,                 # 7 квантилей: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    'loss': QuantileLoss(),           # quantile loss для распределения
    'learning_rate': 1e-3,
    'reduce_on_plateau_patience': 5,

    # Входные переменные
    'time_varying_known_reals': [     # фичи которые известны заранее
        'hour_of_day', 'day_of_week', # временны́е паттерны
    ],
    'time_varying_unknown_reals': [   # фичи доступные только до момента t
        'log_return', 'volatility', 'volume_ratio',
        'rsi', 'macd_hist', 'bb_pct_b', 'atr_norm',
        'funding_rate', 'oi_change',
        # ... все 50+ фичей
    ],
    'static_categoricals': ['symbol'],  # какой символ торгуем

    # Горизонты
    'max_encoder_length': 120,        # смотрим на 120 свечей назад (10 часов)
    'max_prediction_length': 12,      # предсказываем 12 свечей вперёд (1 час)
}
```

**Выход TFT:**
```python
# Вместо одного числа — получаем распределение на каждый горизонт:
{
    'horizon_1':  {'q02': -0.003, 'q10': -0.001, 'q50': 0.001, 'q90': 0.003, 'q98': 0.005},
    'horizon_6':  {'q02': -0.008, 'q10': -0.003, 'q50': 0.004, 'q90': 0.012, 'q98': 0.018},
    'horizon_12': {'q02': -0.015, 'q10': -0.005, 'q50': 0.007, 'q90': 0.020, 'q98': 0.030},
}

# Из этого извлекаем:
direction = sign(q50)           # направление
confidence = 1 - (q90 - q10)   # узкий интервал = высокая уверенность
risk_reward = q50 / abs(q10)   # потенциал vs риск
```

### 3.2 Модель 2: LightGBM (Gradient Boosting)

**Почему нужен gradient boosting в дополнение к TFT:**
- Работает на табличных данных БЕЗ временно́й структуры — ловит другие паттерны
- Очень быстрый inference (<1ms)
- Устойчив к выбросам и шуму
- На daily+ данных часто побеждает нейросети
- Даёт feature importance — понимаем что работает

```python
import lightgbm as lgb

# Features: те же 50+ фичей, но как плоский вектор (без lookback окна)
# Target: direction class (+1, 0, -1) через N свечей

lgb_config = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 500,
    'early_stopping_rounds': 50,
}

# Выход: вероятности трёх классов
# [P(down), P(flat), P(up)] = [0.15, 0.25, 0.60]
```

### 3.3 Модель 3: TCN (Temporal Convolutional Network)

**Почему TCN:**
- Быстрее LSTM/Transformer при inference (параллельные свёртки)
- Dilated convolutions покрывают большой lookback эффективно
- Хорошо ловит локальные паттерны (свечные формации)
- Ортогонален TFT и LightGBM → хорошая диверсификация ансамбля

```python
class TCNClassifier(nn.Module):
    """Temporal Convolutional Network для классификации направления"""

    def __init__(self, input_size=50, num_channels=[64, 64, 32],
                 kernel_size=3, dropout=0.2, num_classes=3):
        super().__init__()
        # Dilated causal convolutions
        # Receptive field = 1 + 2*(kernel_size-1) * sum(2^i for i in range(len(channels)))
        # = 1 + 2*2*(1+2+4) = 29 с 3 слоями
        self.tcn = TemporalBlock(input_size, num_channels, kernel_size, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        tcn_out = self.tcn(x.transpose(1, 2))  # TCN expects (batch, channels, seq)
        last = tcn_out[:, :, -1]                # берём последний timestep
        return self.classifier(last)            # (batch, 3) — logits классов
```

---

## Фаза 4: Meta-Learner & Trade Decision Engine

### 4.1 Объединение моделей

```python
class MetaLearner:
    """
    Взвешенное голосование трёх моделей.
    Веса адаптируются на основе последних N предсказаний каждой модели.
    """

    def __init__(self, window=100):
        self.window = window
        self.model_scores = {
            'tft': deque(maxlen=window),
            'lgbm': deque(maxlen=window),
            'tcn': deque(maxlen=window),
        }

    def combine_predictions(self, tft_pred, lgbm_pred, tcn_pred, regime):
        """
        tft_pred:  {'direction': 1, 'confidence': 0.72, 'q50': 0.004}
        lgbm_pred: {'probs': [0.15, 0.25, 0.60], 'direction': 1}
        tcn_pred:  {'probs': [0.10, 0.30, 0.60], 'direction': 1}
        """
        # Веса на основе recent accuracy
        weights = self._compute_weights()

        # Средневзвешенная уверенность
        tft_conf = tft_pred['confidence']
        lgbm_conf = max(lgbm_pred['probs'])  # макс вероятность класса
        tcn_conf = max(tcn_pred['probs'])

        combined_confidence = (
            weights['tft'] * tft_conf +
            weights['lgbm'] * lgbm_conf +
            weights['tcn'] * tcn_conf
        )

        # Голосование за направление (majority vote с весами)
        votes = {1: 0.0, -1: 0.0, 0: 0.0}
        for model, pred, w in [
            ('tft', tft_pred['direction'], weights['tft']),
            ('lgbm', lgbm_pred['direction'], weights['lgbm']),
            ('tcn', tcn_pred['direction'], weights['tcn']),
        ]:
            votes[pred] += w

        final_direction = max(votes, key=votes.get)
        agreement = votes[final_direction]  # степень согласия моделей

        return {
            'direction': final_direction,
            'confidence': combined_confidence,
            'agreement': agreement,       # 1.0 = все три согласны
            'expected_return': tft_pred['q50'],
            'risk': abs(tft_pred.get('q10', 0)),
        }

    def _compute_weights(self):
        """Веса по Softmax от accuracy за последние N предсказаний"""
        accuracies = {}
        for model, scores in self.model_scores.items():
            if len(scores) < 10:
                accuracies[model] = 1/3  # равные веса пока мало данных
            else:
                accuracies[model] = sum(scores) / len(scores)

        # Softmax с температурой
        temp = 2.0
        exp_scores = {k: np.exp(v / temp) for k, v in accuracies.items()}
        total = sum(exp_scores.values())
        return {k: v / total for k, v in exp_scores.items()}
```

### 4.2 Trade Decision Engine

```python
class TradeDecisionEngine:
    """
    Финальное решение о входе в сделку.
    Учитывает: сигнал моделей + режим рынка + риск-менеджмент.
    """

    def should_trade(self, meta_signal, regime, regime_params, portfolio_state):
        """
        Возвращает: {'action': 'long'/'short'/'hold', 'size_multiplier': 0.0-1.5}
        """
        # 1. Режим запрещает торговлю?
        if not regime_params['trade']:
            return {'action': 'hold', 'reason': f'regime={regime}'}

        # 2. Уверенность ниже порога режима?
        if meta_signal['confidence'] < regime_params['min_confidence']:
            return {'action': 'hold', 'reason': 'low_confidence'}

        # 3. Модели не согласны?
        if meta_signal['agreement'] < 0.5:
            return {'action': 'hold', 'reason': 'disagreement'}

        # 4. Direction = 0 (flat)?
        if meta_signal['direction'] == 0:
            return {'action': 'hold', 'reason': 'flat_signal'}

        # 5. Risk/Reward плохой?
        if meta_signal['risk'] > 0 and meta_signal['expected_return'] / meta_signal['risk'] < 1.5:
            return {'action': 'hold', 'reason': 'bad_risk_reward'}

        # 6. Направление согласуется с трендом режима?
        preferred = regime_params['prefer_direction']
        direction = 'long' if meta_signal['direction'] == 1 else 'short'

        if preferred and direction != preferred:
            # Контр-трендовая сделка — уменьшаем позицию
            size_mult = regime_params['position_scale'] * 0.5
        else:
            size_mult = regime_params['position_scale']

        # 7. Масштабируем по уверенности
        size_mult *= meta_signal['confidence']

        return {
            'action': direction,
            'size_multiplier': size_mult,
            'confidence': meta_signal['confidence'],
            'regime': regime,
            'expected_return': meta_signal['expected_return'],
        }
```

---

## Фаза 5: Training Pipeline (train_pipeline.py)

### 5.1 Walk-Forward Validation

```
Вместо одного train/val split → скользящее окно:

Окно 1: [=====TRAIN=====][=VAL=][embargo]
Окно 2:    [=====TRAIN=====][=VAL=][embargo]
Окно 3:       [=====TRAIN=====][=VAL=][embargo]
...
Окно N:                   [=====TRAIN=====][=VAL=][embargo]

- Train window: 5000 свечей (~17 дней при 5m)
- Val window: 500 свечей (~1.7 дней)
- Embargo: 20 свечей (100 минут) — предотвращает data leakage
- Шаг: 500 свечей
- Итого: ~15-20 фолдов на 10000 свечей

Модель проходит валидацию на ВСЕХ фолдах.
Если средняя метрика по фолдам хуже текущей боевой модели — откат.
```

### 5.2 Расписание обучения

```
# КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: НЕ online learning на каждой свече!

Расписание:
├── Каждые 15 минут: inference (предсказание)
├── Каждые 6 часов:  quick retrain (1 epoch, последние 5000 свечей)
├── Каждые 24 часа:  full retrain (walk-forward CV, all data)
├── Каждые 24 часа:  переобучение Regime Detector
└── Еженедельно:     full ensemble recalibration + meta-learner update

Quick retrain (каждые 6ч):
- Только fine-tune последнего слоя TFT
- Полный retrain LightGBM (он быстрый)
- TCN: fine-tune 1 epoch

Full retrain (каждые 24ч):
- Walk-forward CV на всех данных
- Полное обучение всех трёх моделей
- Сравнение с текущей боевой моделью
- Деплой только если лучше (A/B тест на валидации)
```

### 5.3 Метрики оценки (НЕ только loss)

```python
# Метрики для оценки моделей:

# 1. Direction Accuracy — % верных предсказаний направления
direction_accuracy = correct_direction / total_predictions

# 2. Profitable Signal Rate — % сигналов, которые привели бы к прибыли
profitable_signals = profitable_trades / total_signals

# 3. Risk-Adjusted Return (Sharpe-like)
sharpe = mean(signal_returns) / std(signal_returns)

# 4. Maximum Drawdown от сигналов модели
max_dd = max_drawdown(cumulative_signal_returns)

# 5. Calibration — совпадает ли confidence с реальной accuracy
# Если модель говорит "80% уверен" — она должна быть права ~80%
calibration_error = mean(|predicted_confidence - actual_accuracy|)

# КРИТЕРИЙ ДЕПЛОЯ:
# Новая модель деплоится ТОЛЬКО если:
# - direction_accuracy > 52% на walk-forward CV
# - sharpe > 0.5
# - calibration_error < 0.1
# - profitable_signal_rate > 45%
```

---

## Фаза 6: Файловая структура нового проекта

```
bybit-nn-trader/
├── config.py                    # Конфигурация (обновить)
├── main.py                      # Точка входа (существенно переписать)
│
├── data/
│   ├── data_fetcher.py          # Загрузка OHLCV (обновить)
│   ├── feature_engine.py        # [НОВЫЙ] 50+ фичей, 4 категории
│   ├── market_data.py           # [НОВЫЙ] Funding rate, OI, liquidations
│   └── normalizer.py            # [НОВЫЙ] Expanding window normalization
│
├── models/
│   ├── tft_model.py             # [НОВЫЙ] Temporal Fusion Transformer
│   ├── lgbm_model.py            # [НОВЫЙ] LightGBM classifier
│   ├── tcn_model.py             # [НОВЫЙ] Temporal Convolutional Network
│   ├── regime_detector.py       # [НОВЫЙ] HMM regime detection
│   ├── meta_learner.py          # [НОВЫЙ] Ensemble combination
│   └── model_registry.py        # [НОВЫЙ] Версионирование + A/B тесты
│
├── training/
│   ├── train_pipeline.py        # [НОВЫЙ] Walk-forward CV
│   ├── train_scheduler.py       # [НОВЫЙ] Расписание обучения (6ч/24ч)
│   ├── metrics.py               # [НОВЫЙ] Direction accuracy, Sharpe, calibration
│   └── backtest.py              # [НОВЫЙ] Бэктест сигналов модели
│
├── trading/
│   ├── trader.py                # Исполнение ордеров (обновить)
│   ├── risk_manager.py          # Риск-менеджмент (обновить под режимы)
│   ├── decision_engine.py       # [НОВЫЙ] Финальное решение о сделке
│   └── trade_journal.py         # Журнал сделок (обновить)
│
├── monitoring/
│   ├── training_monitor.py      # Мониторинг обучения (обновить)
│   ├── telegram_bot.py          # Telegram бот (обновить)
│   └── dashboard.py             # [НОВЫЙ] Веб-дашборд (опционально)
│
├── tests/                       # [НОВЫЙ] Тесты!
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_backtest.py
│   └── test_regime.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt             # Обновить: pytorch-forecasting, lightgbm, hmmlearn
└── .env.example
```

---

## Фаза 7: Порядок реализации (приоритеты)

### Этап 1: Фундамент (1-2 дня)
```
1. feature_engine.py — все 50+ фичей
2. normalizer.py — expanding window без leakage
3. Изменить target на log returns + 3-class direction
4. tests/test_features.py — проверка корректности фичей
```

### Этап 2: Regime Detection (0.5 дня)
```
5. regime_detector.py — HMM с 4 состояниями
6. Интеграция в main.py — адаптивные параметры
7. tests/test_regime.py
```

### Этап 3: Core Models (2-3 дня)
```
8. tft_model.py — TFT через pytorch-forecasting
9. lgbm_model.py — LightGBM классификатор
10. tcn_model.py — Temporal Convolutional Network
11. Walk-forward CV для каждой модели
12. tests/test_models.py
```

### Этап 4: Ensemble & Decision Engine (1 день)
```
13. meta_learner.py — взвешенное голосование
14. decision_engine.py — финальная логика торговли
15. model_registry.py — версионирование, A/B
```

### Этап 5: Training Pipeline (1 день)
```
16. train_pipeline.py — walk-forward CV
17. train_scheduler.py — расписание 6ч/24ч
18. metrics.py — direction accuracy, Sharpe, calibration
```

### Этап 6: Backtest & Интеграция (1-2 дня)
```
19. backtest.py — прогон сигналов по историческим данным
20. Обновить main.py — новый flow
21. Обновить risk_manager.py — режим-aware sizing
22. Обновить telegram_bot.py — новые метрики
23. Обновить config.py, requirements.txt, Dockerfile
```

### Этап 7: Тестирование (1 день)
```
24. Полный прогон бэктеста на 30+ днях
25. Paper trading 24-48 часов
26. Анализ метрик, тюнинг порогов
27. Deploy
```

**Итого: ~8-10 дней разработки**

---

## Ключевые зависимости (requirements.txt дополнение)

```
# Новые зависимости:
pytorch-forecasting>=1.0.0    # TFT и TimeSeriesDataSet
pytorch-lightning>=2.0.0      # бэкенд для pytorch-forecasting
lightgbm>=4.0.0               # gradient boosting
hmmlearn>=0.3.0               # Hidden Markov Model для режимов
scikit-learn>=1.3.0           # метрики, preprocessing
optuna>=3.0.0                 # гиперпараметры (опционально)
```

---

## Ожидаемый результат

**Текущая модель:**
- Direction accuracy: ~50% (случайное угадывание)
- Торгует всегда, в любом режиме
- Одна модель, одно число, никакой уверенности
- Переобучается → хаотичные сигналы

**Новая модель (target):**
- Direction accuracy: >55% (каждый % сверх 50% = прибыль)
- Торгует только в понятных режимах с высокой уверенностью
- 3 модели голосуют, дают уверенность и распределение
- Торгует реже, но точнее
- Знает когда НЕ торговать (это важнее чем знать когда торговать)

**Главный принцип новой модели:**
> Лучшая сделка — та, которую мы НЕ открыли когда были не уверены.
