import pandas as pd
from loguru import logger
from config import (
    RISK_PER_TRADE,
    FRACTIONAL_KELLY,
    USE_TRAILING,
    TRAILING_STOP_PCT,
    SL_ATR_MULTIPLIER,
    TP_ATR_MULTIPLIER,
)
from data_fetcher import exchange  # REST клиент


def calculate_kelly(win_rate: float = 0.52, avg_win: float = 0.015, avg_loss: float = 0.01) -> float:
    """Расчёт Fractional Kelly"""
    if avg_loss == 0:
        return 0.0
    b = avg_win / abs(avg_loss)
    q = 1 - win_rate
    kelly = (b * win_rate - q) / b
    return max(0.0, min(1.0, kelly * FRACTIONAL_KELLY))


def calculate_position_size(balance: float, current_price: float, atr: float = None) -> float:
    """Динамический размер позиции с Fractional Kelly и ATR"""
    risk_amount = balance * RISK_PER_TRADE
    kelly_fraction = calculate_kelly()
    risk_amount *= kelly_fraction

    if atr is None or atr <= 0:
        size = risk_amount / current_price
    else:
        stop_distance = atr * SL_ATR_MULTIPLIER
        size = risk_amount / stop_distance

    # Защита: не больше 25% баланса на одну позицию
    size = min(size, balance * 0.25)
    return max(size, 0.0)  # не отрицательный размер


async def set_tp_sl_trailing(symbol: str, side: str, entry_price: float, atr: float):
    """
    Устанавливает Stop-Loss, Take-Profit и Trailing Stop для позиции на Bybit V5.
    Работает через exchange.set_trading_stop()
    """
    try:
        # Очищаем символ (Bybit принимает BTCUSDT, а не BTCUSDT/USDT:USDT)
        clean_symbol = symbol.split('/')[0] if '/' in symbol else symbol

        # Расчёт цен
        if side.lower() == 'buy':  # Long
            sl_price = round(entry_price * (1 - SL_ATR_MULTIPLIER * 0.01), 2)
            tp_price = round(entry_price * (1 + TP_ATR_MULTIPLIER * 0.01), 2)
        else:  # Short
            sl_price = round(entry_price * (1 + SL_ATR_MULTIPLIER * 0.01), 2)
            tp_price = round(entry_price * (1 - TP_ATR_MULTIPLIER * 0.01), 2)

        params = {
            "category": "linear",
            "symbol": clean_symbol,
            "stopLoss": str(sl_price),
            "takeProfit": str(tp_price),
            "slTriggerBy": "MarkPrice",
            "tpTriggerBy": "MarkPrice",
            "tpslMode": "Full",  # Full = вся позиция
        }

        if USE_TRAILING and TRAILING_STOP_PCT > 0:
            params["trailingStop"] = str(TRAILING_STOP_PCT)  # в процентах

        # Основной вызов через CCXT
        result = exchange.set_trading_stop(params)

        logger.success(
            f"✅ TP/SL{' + TrailingStop {TRAILING_STOP_PCT}%' if USE_TRAILING else ''} "
            f"установлены для {side.upper()} {clean_symbol} | "
            f"SL: {sl_price} | TP: {tp_price}"
        )
        return result

    except Exception as e:
        logger.error(f"❌ Ошибка установки TP/SL/Trailing для {symbol}: {e}")
        # Попробуем вывести больше деталей для отладки
        logger.debug(f"Параметры: {params}")
        raise