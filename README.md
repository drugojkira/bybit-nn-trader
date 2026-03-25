# Bybit NN Trader v4

Торговый бот на основе нейронной сети для Binance Futures с реал-тайм WebSocket, адаптивным риск-менеджментом и Telegram-уведомлениями.

---

## Возможности

- **Нейронная сеть** — LSTM/Transformer модель с онлайн дообучением на каждой новой свече
- **WebSocket** — реал-тайм получение свечей через ccxt.pro, exponential backoff + circuit breaker
- **Технические индикаторы** — RSI, MACD, Bollinger Bands, ATR, Volume Ratio
- **Адаптивный порог сигнала** — порог входа масштабируется по ATR (0.1%–1.0%)
- **Риск-менеджмент** — Fractional Kelly, ATR-based SL/TP, Trailing Stop, лимит просадки
- **Telegram-уведомления** — сигналы, отчёты каждые 50 свечей, алерты о просадке
- **Журнал сделок** — статистика win rate, PnL, drawdown по каждому символу
- **REST API** — эндпоинты для мониторинга состояния бота
- **Docker** — готовый `docker-compose.yml` с healthcheck

---

## Структура проекта

```
├── main.py            # FastAPI приложение, основной цикл
├── model.py           # Нейронная сеть, обучение, предсказание
├── data_fetcher.py    # REST + WebSocket клиент Binance (ccxt/ccxt.pro)
├── trader.py          # Открытие/закрытие позиций с retry
├── risk_manager.py    # Kelly, ATR-позиционирование, TP/SL, drawdown
├── trade_journal.py   # Журнал сделок и статистика
├── telegram_bot.py    # Telegram уведомления
├── config.py          # Загрузка переменных окружения
├── models/            # Сохранённые веса моделей (.pth)
├── .env               # Ключи API и параметры (не коммитить!)
├── .env.example       # Шаблон конфига
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Быстрый старт

### 1. Клонировать и настроить окружение

```bash
git clone <repo>
cd bybit-nn-trader

python3.12 -m venv venv312
source venv312/bin/activate   # Windows: venv312\Scripts\activate
pip install -r requirements.txt
```

### 2. Настроить `.env`

```bash
cp .env.example .env
```

Заполнить:

| Переменная | Описание |
|---|---|
| `BINANCE_API_KEY` | API ключ Binance |
| `BINANCE_API_SECRET` | Secret Binance |
| `TESTNET` | `true` для тестнета, `false` для реальной торговли |
| `TELEGRAM_TOKEN` | Токен бота от @BotFather |
| `TELEGRAM_CHAT_ID` | Ваш chat ID (узнать у @userinfobot) |
| `SYMBOLS` | Символы через запятую: `BTCUSDT,ETHUSDT` |
| `TIMEFRAME` | Таймфрейм: `1m`, `5m`, `15m`, `1h` |
| `RISK_PER_TRADE` | Риск на сделку (0.01 = 1% баланса) |
| `FRACTIONAL_KELLY` | Доля Kelly (0.25–0.75, меньше = консервативнее) |
| `MAX_DRAWDOWN` | Максимальная просадка до остановки (0.10 = 10%) |
| `USE_TRAILING` | Включить trailing stop: `true`/`false` |
| `TRAILING_STOP_PCT` | Размер trailing stop в % |
| `SL_ATR_MULTIPLIER` | Stop Loss = ATR × множитель |
| `TP_ATR_MULTIPLIER` | Take Profit = ATR × множитель |

### 3. Запустить

```bash
python main.py
```

---

## Запуск через Docker

```bash
docker-compose up -d
```

Логи:
```bash
docker-compose logs -f
```

Остановить:
```bash
docker-compose down
```

---

## Как работает бот

```
WebSocket (новая свеча)
        │
        ▼
  Дообучение модели на последних 800 свечах
        │
        ▼
  Предсказание цены следующей свечи
        │
        ▼
  Вычисление сигнала:
    price_change > +threshold → LONG
    price_change < -threshold → SHORT
    иначе → нет сигнала
        │
        ▼
  Проверка лимита просадки
        │
        ▼
  Открытие/закрытие позиции + TP/SL/Trailing Stop
        │
        ▼
  Уведомление в Telegram
```

**Адаптивный порог** рассчитывается как 30% от ATR/цена, с ограничением 0.1%–1.0%. При высокой волатильности порог растёт, фильтруя шум.

**Fractional Kelly** пересчитывается на основе реальной статистики последних 100 сделок (win rate, avg win/loss). Если матожидание отрицательное — бот не открывает сделки.

---

## Мониторинг (REST API)

| Эндпоинт | Описание |
|---|---|
| `GET /health` | Статус бота |
| `GET /stats` | Статистика по всем символам |
| `GET /stats/{symbol}` | Статистика по символу (например: `/stats/BTC`) |

Пример:
```bash
curl http://localhost:8000/stats
```

---

## Риск-менеджмент

- **Stop Loss**: `entry_price ± ATR × SL_ATR_MULTIPLIER` (по умолчанию ATR×2)
- **Take Profit**: `entry_price ± ATR × TP_ATR_MULTIPLIER` (по умолчанию ATR×4)
- **Trailing Stop**: автоматически двигается за ценой на `TRAILING_STOP_PCT`%
- **Drawdown limit**: при достижении `MAX_DRAWDOWN` торговля останавливается, уведомление в Telegram
- **Защита от овербетинга**: не более 25% баланса на одну позицию

---

## Требования

- Python 3.12+
- Binance Futures аккаунт (для тестнета: testnet.binancefuture.com)
- Telegram бот

---

## Важно

> Бот работает на **реальных деньгах** при `TESTNET=false`. Всегда тестируйте на testnet перед запуском с реальным балансом. Автор не несёт ответственности за финансовые потери.
