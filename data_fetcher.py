import ccxt
import ccxt.pro as ccxtpro
import pandas as pd
import asyncio
import time
from loguru import logger
from config import API_KEY, API_SECRET, TESTNET, TIMEFRAME, CCXT_SYMBOLS

# REST клиент
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
})
if TESTNET:
    exchange.set_sandbox_mode(True)


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
        # Удаляем старые записи
        self._calls = [t for t in self._calls if now - t < self.period]

        if len(self._calls) >= self.max_calls:
            wait_time = self.period - (now - self._calls[0])
            if wait_time > 0:
                logger.debug(f"Rate limit: ждём {wait_time:.2f}с")
                await asyncio.sleep(wait_time)

        self._calls.append(time.monotonic())


# Глобальный лимитер: 10 запросов/секунду (Binance лимит = 20/с, берём с запасом)
rate_limiter = RateLimiter(max_calls=10, period=1.0)


async def fetch_ohlcv_async(symbol: str, limit=1000) -> pd.DataFrame:
    """Асинхронная версия fetch_ohlcv с rate limiting"""
    await rate_limiter.acquire()
    return fetch_ohlcv(symbol, limit)


def fetch_ohlcv(symbol: str, limit=1000) -> pd.DataFrame:
    """REST API для загрузки исторических свечей"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except ccxt.RateLimitExceeded as e:
        logger.warning(f"Rate limit от биржи для {symbol}, ждём 5 секунд: {e}")
        import time as _time
        _time.sleep(5)
        return fetch_ohlcv(symbol, limit)
    except ccxt.NetworkError as e:
        logger.error(f"Сетевая ошибка fetch_ohlcv {symbol}: {e}")
        raise
    except ccxt.ExchangeError as e:
        logger.error(f"Ошибка биржи fetch_ohlcv {symbol}: {e}")
        raise


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


# === WebSocket с exponential backoff и circuit breaker ===

class CircuitBreaker:
    """
    Circuit breaker: если слишком много ошибок подряд — переходим в 'open' state,
    ждём длительную паузу перед повторной попыткой.
    """

    def __init__(self, max_failures: int = 5, reset_timeout: float = 300.0):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout  # секунд
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = 'closed'  # closed / open / half_open

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
        """Возвращает True если можно продолжать, иначе ждёт"""
        if self.state == 'closed':
            return True

        elapsed = time.monotonic() - self.last_failure_time
        if elapsed >= self.reset_timeout:
            self.state = 'half_open'
            logger.info("Circuit breaker → HALF_OPEN: пробуем переподключиться")
            return True

        wait = self.reset_timeout - elapsed
        logger.info(f"Circuit breaker OPEN: ждём ещё {wait:.0f}с")
        await asyncio.sleep(min(wait, 30))  # проверяем каждые 30с
        return False


async def watch_ohlcv_forever(symbol: str, callback):
    """
    Бесконечный WebSocket для одного символа.
    Exponential backoff + Circuit breaker.
    """
    ws_exchange = None
    backoff = 5  # стартовый backoff в секундах
    max_backoff = 300  # максимум 5 минут
    circuit = CircuitBreaker(max_failures=5, reset_timeout=300)

    while True:
        # Проверяем circuit breaker
        if not await circuit.check():
            continue

        try:
            ws_exchange = ccxtpro.binance({
                'apiKey': API_KEY,
                'secret': API_SECRET,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'},
            })
            if TESTNET:
                ws_exchange.set_sandbox_mode(True)

            logger.info(f"WebSocket запущен для {symbol}")
            backoff = 5  # сброс backoff при успешном подключении

            while True:
                ohlcv = await ws_exchange.watch_ohlcv(symbol, TIMEFRAME)
                df = pd.DataFrame(
                    ohlcv[-200:],
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                circuit.record_success()
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
            backoff = min(backoff * 2, max_backoff)  # exponential backoff
