from loguru import logger
from data_fetcher import exchange, get_atr
from risk_manager import calculate_position_size, set_tp_sl_trailing   # предполагаем, что функция асинхронная

def get_current_position(symbol: str):
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
    pos_side = get_current_position(symbol)
    if not pos_side:
        return
    side = 'sell' if pos_side == 'long' else 'buy'
    try:
        positions = exchange.fetch_positions([symbol])
        amount = abs(float(positions[0]['contracts']))
        order = exchange.create_order(symbol, 'market', side, amount, params={'reduceOnly': True})
        logger.success(f"Закрыта позиция {side.upper()} {amount} {symbol}")
    except Exception as e:
        logger.error(f"Ошибка закрытия {symbol}: {e}")

async def place_order_with_risk(symbol: str, side: str, model):
    try:
        df = fetch_ohlcv(symbol, limit=300)   # для ATR
        atr = get_atr(df)
        balance = exchange.fetch_balance().get('USDT', {}).get('free', 0)
        current_price = df['close'].iloc[-1]

        amount = calculate_position_size(balance, current_price, atr)

        if amount <= 0:
            return None

        order = exchange.create_order(symbol, 'market', side, amount)
        logger.success(f"Открыт {side.upper()} {amount:.6f} {symbol} @ {current_price:.2f}")

        await set_tp_sl_trailing(symbol, side, current_price, atr)
        return order
    except Exception as e:
        logger.error(f"Ошибка открытия {symbol}: {e}")
        return None