import asyncio
import ccxt
from loguru import logger
from data_fetcher import exchange, spot_exchange, get_atr, fetch_ohlcv_async
from risk_manager import calculate_position_size, set_tp_sl_trailing
from trade_journal import journal
from config import CCXT_SYMBOLS


MAX_RETRIES = 3
RETRY_DELAY = 2.0


async def retry_exchange_call(func, *args, retries=MAX_RETRIES, **kwargs):
    """Обёртка для sync-вызовов биржи через executor с retry"""
    loop = asyncio.get_event_loop()
    for attempt in range(retries):
        try:
            result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            return result
        except ccxt.RateLimitExceeded as e:
            wait = RETRY_DELAY * (2 ** attempt)
            logger.warning(f"Rate limit (попытка {attempt + 1}/{retries}), ждём {wait}с: {e}")
            await asyncio.sleep(wait)
        except ccxt.NetworkError as e:
            wait = RETRY_DELAY * (2 ** attempt)
            logger.warning(f"Сетевая ошибка (попытка {attempt + 1}/{retries}), ждём {wait}с: {e}")
            await asyncio.sleep(wait)
        except ccxt.ExchangeError as e:
            logger.error(f"Ошибка биржи (не повторяем): {e}")
            raise
    raise ccxt.NetworkError(f"Исчерпаны попытки ({retries}) для {func.__name__}")


# === Получение позиций ===

async def get_current_position(symbol: str):
    """Получение стороны текущей позиции по символу"""
    try:
        positions = await retry_exchange_call(exchange.fetch_positions, [symbol])
        for pos in positions:
            if float(pos.get('contracts') or 0) != 0:
                return pos['side'].lower()
        return None
    except Exception as e:
        logger.error(f"Ошибка получения позиции {symbol}: {e}")
        return None


async def get_position_info(symbol: str) -> dict | None:
    """Полная информация о позиции по символу"""
    try:
        positions = await retry_exchange_call(exchange.fetch_positions, [symbol])
        for pos in positions:
            contracts = float(pos.get('contracts') or 0)
            if contracts != 0:
                return {
                    'symbol': symbol,
                    'side': pos['side'].lower(),
                    'contracts': contracts,
                    'entry_price': float(pos.get('entryPrice') or 0),
                    'mark_price': float(pos.get('markPrice') or 0),
                    'unrealized_pnl': float(pos.get('unrealizedPnl') or 0),
                    'leverage': int(pos.get('leverage') or 1),
                    'liquidation_price': float(pos.get('liquidationPrice') or 0),
                }
        return None
    except Exception as e:
        logger.error(f"Ошибка получения позиции {symbol}: {e}")
        return None


async def get_all_positions() -> list[dict]:
    """
    Получение ВСЕХ открытых позиций по всем символам.
    Возвращает список dict с информацией о каждой позиции.
    """
    all_pos = []
    try:
        positions = await retry_exchange_call(exchange.fetch_positions, CCXT_SYMBOLS)
        for pos in positions:
            contracts = float(pos.get('contracts') or 0)
            if contracts != 0:
                all_pos.append({
                    'symbol': pos.get('symbol', ''),
                    'side': pos['side'].lower(),
                    'contracts': contracts,
                    'entry_price': float(pos.get('entryPrice') or 0),
                    'mark_price': float(pos.get('markPrice') or 0),
                    'unrealized_pnl': float(pos.get('unrealizedPnl') or 0),
                    'leverage': int(pos.get('leverage') or 1),
                })
    except Exception as e:
        logger.error(f"Ошибка получения всех позиций: {e}")
    return all_pos


# === Балансы ===

async def get_balance() -> dict:
    """Баланс фьючерсного аккаунта"""
    try:
        balance_info = await retry_exchange_call(exchange.fetch_balance)
        usdt = balance_info.get('USDT', {})
        return {
            'free': float(usdt.get('free', 0)),
            'used': float(usdt.get('used', 0)),
            'total': float(usdt.get('total', 0)),
        }
    except Exception as e:
        logger.error(f"Ошибка получения баланса: {e}")
        return {'free': 0, 'used': 0, 'total': 0}


async def get_spot_balance() -> dict:
    """Баланс спот аккаунта"""
    try:
        balance_info = await retry_exchange_call(spot_exchange.fetch_balance)
        usdt = balance_info.get('USDT', {})
        return {
            'free': float(usdt.get('free', 0)),
            'used': float(usdt.get('used', 0)),
            'total': float(usdt.get('total', 0)),
        }
    except Exception as e:
        logger.error(f"Ошибка получения спот баланса: {e}")
        return {'free': 0, 'used': 0, 'total': 0}


# === Торговля ===

async def close_position(symbol: str):
    """Закрытие позиции по символу"""
    pos_side = await get_current_position(symbol)
    if not pos_side:
        return

    side = 'sell' if pos_side == 'long' else 'buy'
    try:
        positions = await retry_exchange_call(exchange.fetch_positions, [symbol])
        amount = abs(float(positions[0]['contracts']))
        if amount == 0:
            return

        current_price = float(positions[0].get('markPrice', 0))

        await retry_exchange_call(
            exchange.create_order,
            symbol, 'market', side, amount,
            params={'reduceOnly': True}
        )
        logger.success(f"Закрыта позиция {pos_side.upper()} {amount} {symbol}")

        if journal.has_open_trade(symbol):
            journal.close_trade(symbol, current_price, exit_reason="signal_change")

    except Exception as e:
        logger.error(f"Ошибка закрытия {symbol}: {e}")


async def place_order_with_risk(symbol: str, side: str, model, prediction: float = None):
    """Открытие позиции с риск-менеджментом"""
    try:
        df = await fetch_ohlcv_async(symbol, limit=300)
        atr = get_atr(df)
        current_price = df['close'].iloc[-1]

        balance_info = await get_balance()
        balance = balance_info['free']

        if balance <= 0:
            logger.warning(f"Нулевой баланс, пропускаем ордер для {symbol}")
            return None

        amount = calculate_position_size(balance, current_price, atr, symbol)

        if amount <= 0:
            logger.info(f"Размер позиции = 0 для {symbol}, пропускаем")
            return None

        # Проверка минимального размера ордера
        try:
            markets = exchange.markets
            if symbol in markets:
                min_amount = markets[symbol].get('limits', {}).get('amount', {}).get('min', 0)
                if min_amount and amount < min_amount:
                    amount = min_amount
        except Exception:
            pass

        order = await retry_exchange_call(
            exchange.create_order,
            symbol, 'market', side, amount
        )
        logger.success(f"Открыт {side.upper()} {amount:.6f} {symbol} @ {current_price:.2f}")

        journal.open_trade(
            symbol=symbol,
            side=side,
            entry_price=current_price,
            amount=amount,
            prediction=prediction,
        )

        await set_tp_sl_trailing(symbol, side, current_price, atr)
        return order

    except Exception as e:
        logger.error(f"Ошибка открытия {symbol}: {e}")
        return None
