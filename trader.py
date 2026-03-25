import asyncio
import ccxt
from loguru import logger
from data_fetcher import exchange, get_atr, fetch_ohlcv
from risk_manager import calculate_position_size, set_tp_sl_trailing
from trade_journal import journal


MAX_RETRIES = 3
RETRY_DELAY = 2.0  # секунд


async def retry_exchange_call(func, *args, retries=MAX_RETRIES, **kwargs):
    """
    Обёртка для вызовов биржи с retry и exponential backoff.
    Повторяет при сетевых ошибках и rate limit, но не при ошибках логики.
    """
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except ccxt.RateLimitExceeded as e:
            wait = RETRY_DELAY * (2 ** attempt)
            logger.warning(f"Rate limit (попытка {attempt + 1}/{retries}), ждём {wait}с: {e}")
            await asyncio.sleep(wait)
        except ccxt.NetworkError as e:
            wait = RETRY_DELAY * (2 ** attempt)
            logger.warning(f"Сетевая ошибка (попытка {attempt + 1}/{retries}), ждём {wait}с: {e}")
            await asyncio.sleep(wait)
        except ccxt.ExchangeError as e:
            # Ошибки биржи (недостаточно средств, неверный символ и т.д.) — не повторяем
            logger.error(f"Ошибка биржи (не повторяем): {e}")
            raise
    raise ccxt.NetworkError(f"Исчерпаны попытки ({retries}) для {func.__name__}")


def get_current_position(symbol: str):
    """Получение текущей позиции с обработкой ошибок"""
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if float(pos.get('contracts', 0)) != 0:
                return pos['side'].lower()
        return None
    except Exception as e:
        logger.error(f"Ошибка получения позиции {symbol}: {e}")
        return None


async def close_position(symbol: str):
    """Закрытие позиции с retry и записью в журнал"""
    pos_side = get_current_position(symbol)
    if not pos_side:
        return

    side = 'sell' if pos_side == 'long' else 'buy'
    try:
        positions = await retry_exchange_call(exchange.fetch_positions, [symbol])
        amount = abs(float(positions[0]['contracts']))
        if amount == 0:
            return

        # Получаем текущую цену для журнала
        current_price = float(positions[0].get('markPrice', 0))

        order = await retry_exchange_call(
            exchange.create_order,
            symbol, 'market', side, amount,
            params={'reduceOnly': True}
        )
        logger.success(f"Закрыта позиция {pos_side.upper()} {amount} {symbol}")

        # Записываем закрытие в журнал
        if journal.has_open_trade(symbol):
            journal.close_trade(symbol, current_price, exit_reason="signal_change")

    except Exception as e:
        logger.error(f"Ошибка закрытия {symbol}: {e}")


async def place_order_with_risk(symbol: str, side: str, model, prediction: float = None):
    """Открытие позиции с риск-менеджментом, retry и журналом"""
    try:
        df = fetch_ohlcv(symbol, limit=300)
        atr = get_atr(df)
        current_price = df['close'].iloc[-1]

        # Получение баланса с retry
        balance_info = await retry_exchange_call(exchange.fetch_balance)
        balance = float(balance_info.get('USDT', {}).get('free', 0))

        if balance <= 0:
            logger.warning(f"Нулевой баланс, пропускаем ордер для {symbol}")
            return None

        amount = calculate_position_size(balance, current_price, atr, symbol)

        if amount <= 0:
            logger.info(f"Размер позиции = 0 для {symbol}, пропускаем")
            return None

        order = await retry_exchange_call(
            exchange.create_order,
            symbol, 'market', side, amount
        )
        logger.success(f"Открыт {side.upper()} {amount:.6f} {symbol} @ {current_price:.2f}")

        # Записываем в журнал
        journal.open_trade(
            symbol=symbol,
            side=side,
            entry_price=current_price,
            amount=amount,
            prediction=prediction,
        )

        # Устанавливаем TP/SL
        await set_tp_sl_trailing(symbol, side, current_price, atr)
        return order

    except Exception as e:
        logger.error(f"Ошибка открытия {symbol}: {e}")
        return None
