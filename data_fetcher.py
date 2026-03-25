import ccxt
import ccxt.pro as ccxtpro
import pandas as pd
import asyncio
import time
from loguru import logger
from config import API_KEY, API_SECRET, TIMEFRAME, CCXT_SYMBOLS

# REST клиент фьючерсов (demo-fapi.binance.com)
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
})
exchange.enable_demo_trading(True)

# REST клиент спота (demo-api.binance.com) — для мониторинга спот-баланса
spot_exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
})
spot_exchange.enable_demo_trading(True)


# === Rate Limiter ===

class RateLimiter:
    """Ограничение частоты запросов к API (token bucket)"""

    def __init__(self, max_calls: int = 10, period: float = 1.0):
        self.max_calls = max_calls
        self.period = period
        self._calls: list[float] = []

    async def acquire(self):
        """Ждёт, пока не появится свободный слот"""
        now = time.monotonic()
        self._calls = [t for t in self._calls if now - t < self.period]

        if len(self._calls) >= self.max_calls:
            wait_time = self.period - (now - self._calls[0])
            if wait_time > 0:
                logger.debug(f"Rate limit: ждём {wait_time:.2f}с")
                await asyncio.sleep(wait_time)

        self._calls.append(time.monotonic())


rate_limiter = RateLimiter(max_calls=10, period=1.0)


def fetch_ohlcv(symbol: str, limit=1000) -> pd.DataFrame:
    """
    REST API для загрузки исторических свечей.
    Цикл с retry вместо рекурсии (fix stack overflow).
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except ccxt.RateLimitExceeded as e:
            wait = min(5 * (2 ** attempt), 60)
            logger.warning(
                f"Rate limit для {symbol} (попытка {attempt + 1}/{max_retries}), "
                f"ждём {wait}с: {e}"
            )
            time.sleep(wait)
        except ccxt.NetworkError as e:
            wait = min(3 * (2 ** attempt), 30)
            logger.warning(
                f"Сетевая ошибка fetch_ohlcv {symbol} (попытка {attempt + 1}/{max_retries}), "
                f"ждём {wait}с: {e}"
            )
            time.sleep(wait)
        except ccxt.ExchangeError as e:
            logger.error(f"Ошибка биржи fetch_ohlcv {symbol}: {e}")
            raise

    raise ccxt.NetworkError(f"Исчерпаны попытки ({max_retries}) fetch_ohlcv для {symbol}")


def fetch_ohlcv_paginated(symbol: str, total_limit=10000, batch_size=1000) -> pd.DataFrame:
    """
    Загружает большой датасет через пагинацию (несколько запросов по batch_size).
    Binance позволяет до 1000 свечей за запрос, используем since для смещения.
    """
    all_ohlcv = []
    since = None
    max_retries = 5

    while len(all_ohlcv) < total_limit:
        fetch_limit = min(batch_size, total_limit - len(all_ohlcv))
        fetched = None

        for attempt in range(max_retries):
            try:
                fetched = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=fetch_limit)
                break
            except ccxt.RateLimitExceeded as e:
                wait = min(5 * (2 ** attempt), 60)
                logger.warning(f"Rate limit (пагинация {symbol}), ждём {wait}с: {e}")
                time.sleep(wait)
            except ccxt.NetworkError as e:
                wait = min(3 * (2 ** attempt), 30)
                logger.warning(f"Сетевая ошибка (пагинация {symbol}), ждём {wait}с: {e}")
                time.sleep(wait)
            except ccxt.ExchangeError as e:
                logger.error(f"Ошибка биржи fetch_ohlcv_paginated {symbol}: {e}")
                raise

        if not fetched:
            break

        if since is None:
            # Первый запрос — берём самые старые свечи для нужного периода
            # Вычисляем since как начало нужного диапазона
            timeframe_ms = exchange.parse_timeframe(TIMEFRAME) * 1000
            since_start = exchange.milliseconds() - total_limit * timeframe_ms
            fetched = None
            for attempt in range(max_retries):
                try:
                    fetched = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since_start, limit=fetch_limit)
                    break
                except Exception as e:
                    time.sleep(3)
            if not fetched:
                break
            all_ohlcv.extend(fetched)
            if len(fetched) < fetch_limit:
                break
            since = fetched[-1][0] + 1
        else:
            all_ohlcv.extend(fetched)
            if len(fetched) < fetch_limit:
                break
            since = fetched[-1][0] + 1

        logger.info(f"Пагинация {symbol}: загружено {len(all_ohlcv)}/{total_limit} свечей")
        time.sleep(0.2)  # небольшая пауза между запросами

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Загружено {len(df)} свечей для {symbol} (пагинация)")
    return df


async def fetch_ohlcv_async(symbol: str, limit=1000) -> pd.DataFrame:
    """Асинхронная обёртка — выполняет sync fetch_ohlcv в executor"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fetch_ohlcv, symbol, limit)


def get_atr(df: pd.DataFrame, period=14) -> float:
    """Average True Range"""
    if len(df) < period + 1:
        logger.warning(f"Недостаточно данных для ATR: {len(df)} < {period + 1}")
        return 0.0
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


# === WebSocket с буфером свечей, exponential backoff и circuit breaker ===

class CandleBuffer:
    """
    Буфер свечей для WebSocket.
    Предзагружает историю через REST и дополняет из WebSocket,
    чтобы всегда иметь достаточно данных для обучения.
    """

    def __init__(self, symbol: str, max_size: int = 1500):
        self.symbol = symbol
        self.max_size = max_size
        self.df: pd.DataFrame | None = None
        self._initialized = False

    def initialize(self, historical_df: pd.DataFrame):
        """Загружает начальную историю"""
        self.df = historical_df.copy().reset_index(drop=True)
        self._initialized = True
        logger.info(f"CandleBuffer [{self.symbol}]: загружено {len(self.df)} исторических свечей")

    def update(self, ws_candles: list) -> pd.DataFrame:
        """
        Обновляет буфер новыми свечами из WebSocket.
        Возвращает полный DataFrame с историей.
        """
        if not self._initialized or self.df is None:
            # Если нет истории — используем только WS данные
            self.df = pd.DataFrame(
                ws_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
            return self.df

        # Конвертируем WS данные
        new_df = pd.DataFrame(
            ws_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')

        if len(new_df) == 0:
            return self.df

        # Берём только свечи новее последней в буфере
        last_ts = self.df['timestamp'].iloc[-1]
        new_candles = new_df[new_df['timestamp'] > last_ts]

        if len(new_candles) > 0:
            self.df = pd.concat([self.df, new_candles], ignore_index=True)

        # Обновляем последнюю свечу (может быть незакрытая)
        if len(new_df) > 0:
            latest_ws = new_df.iloc[-1]
            mask = self.df['timestamp'] == latest_ws['timestamp']
            if mask.any():
                idx = self.df[mask].index[-1]
                self.df.loc[idx] = latest_ws

        # Ограничиваем размер буфера
        if len(self.df) > self.max_size:
            self.df = self.df.iloc[-self.max_size:].reset_index(drop=True)

        return self.df

    @property
    def size(self) -> int:
        return len(self.df) if self.df is not None else 0


class CircuitBreaker:
    """
    Circuit breaker: если слишком много ошибок подряд — переходим в 'open' state.
    """

    def __init__(self, max_failures: int = 5, reset_timeout: float = 300.0):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = 'closed'

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.monotonic()
        if self.failures >= self.max_failures:
            self.state = 'open'
            logger.warning(
                f"Circuit breaker OPEN: {self.failures} ошибок подряд. "
                f"Пауза {self.reset_timeout}с"
            )

    def record_success(self):
        self.failures = 0
        self.state = 'closed'

    async def check(self) -> bool:
        if self.state == 'closed':
            return True

        elapsed = time.monotonic() - self.last_failure_time
        if elapsed >= self.reset_timeout:
            self.state = 'half_open'
            logger.info("Circuit breaker → HALF_OPEN: пробуем переподключиться")
            return True

        wait = self.reset_timeout - elapsed
        logger.info(f"Circuit breaker OPEN: ждём ещё {wait:.0f}с")
        await asyncio.sleep(min(wait, 30))
        return False


# Глобальные буферы свечей
candle_buffers: dict[str, CandleBuffer] = {}


async def watch_ohlcv_forever(symbol: str, callback):
    """
    Бесконечный WebSocket для одного символа.
    Предзагружает историю через REST, далее дополняет из WebSocket.
    """
    ws_exchange = None
    backoff = 5
    max_backoff = 300
    circuit = CircuitBreaker(max_failures=5, reset_timeout=300)

    # Инициализируем буфер свечей с исторической загрузкой
    if symbol not in candle_buffers:
        candle_buffers[symbol] = CandleBuffer(symbol, max_size=10000)

    buffer = candle_buffers[symbol]

    # Предзагружаем историю
    if not buffer._initialized:
        try:
            logger.info(f"Предзагрузка истории для {symbol}...")
            historical = fetch_ohlcv(symbol, limit=5000)
            buffer.initialize(historical)
        except Exception as e:
            logger.error(f"Ошибка предзагрузки истории {symbol}: {e}")

    while True:
        if not await circuit.check():
            continue

        try:
            ws_exchange = ccxtpro.binance({
                'apiKey': API_KEY,
                'secret': API_SECRET,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'},
            })
            ws_exchange.enable_demo_trading(True)

            logger.info(f"WebSocket запущен для {symbol}")
            backoff = 5

            while True:
                ohlcv = await ws_exchange.watch_ohlcv(symbol, TIMEFRAME)

                # Обновляем буфер полным набором WS данных
                df = buffer.update(ohlcv)

                circuit.record_success()
                logger.debug(
                    f"[{symbol}] Буфер: {buffer.size} свечей | "
                    f"WS batch: {len(ohlcv)}"
                )
                await callback(df, symbol)

        except asyncio.CancelledError:
            logger.info(f"WebSocket для {symbol} отменён (shutdown)")
            break

        except Exception as e:
            logger.error(f"WebSocket ошибка для {symbol}: {e}")
            circuit.record_failure()

            if ws_exchange:
                try:
                    await ws_exchange.close()
                except Exception:
                    pass

            logger.info(f"Реконнект {symbol} через {backoff}с...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
