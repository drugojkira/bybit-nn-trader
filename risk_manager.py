import pandas as pd
from loguru import logger
from config import (
    RISK_PER_TRADE,
    FRACTIONAL_KELLY,
    MAX_DRAWDOWN,
    USE_TRAILING,
    TRAILING_STOP_PCT,
    SL_ATR_MULTIPLIER,
    TP_ATR_MULTIPLIER,
)
from data_fetcher import exchange
from trade_journal import journal


def calculate_kelly(symbol: str = None) -> float:
    """
    Расчёт Fractional Kelly на основе реальной статистики сделок.
    Если недостаточно данных — используем консервативные дефолты.
    """
    stats = journal.get_statistics(symbol=symbol, last_n=100)

    win_rate = stats['win_rate']
    avg_win = stats['avg_win']
    avg_loss = stats['avg_loss']

    if avg_loss == 0:
        return 0.0

    b = avg_win / abs(avg_loss)
    q = 1 - win_rate
    kelly = (b * win_rate - q) / b

    result = max(0.0, min(1.0, kelly * FRACTIONAL_KELLY))

    logger.debug(
        f"Kelly [{symbol or 'all'}]: WR={win_rate:.2%}, "
        f"AvgWin={avg_win:.4f}, AvgLoss={avg_loss:.4f}, "
        f"Kelly={kelly:.4f}, Fractional={result:.4f} "
        f"(из {stats['total_trades']} сделок)"
    )
    return result


def calculate_adaptive_threshold(atr: float, current_price: float) -> float:
    """
    Адаптивный порог сигнала на основе ATR.
    Порог = ATR / цена * множитель (минимум 0.1%, максимум 1%)
    """
    if atr is None or atr <= 0 or current_price <= 0:
        return 0.002  # дефолт 0.2%

    # ATR как доля от цены
    atr_pct = atr / current_price

    # Порог = 30% от ATR (достаточно для фильтрации шума, но не слишком высокий)
    threshold = atr_pct * 0.3

    # Ограничения: не менее 0.1%, не более 1%
    threshold = max(0.001, min(0.01, threshold))

    logger.debug(f"Адаптивный порог: {threshold:.4f} (ATR={atr:.2f}, Price={current_price:.2f})")
    return threshold


def check_drawdown_limit(symbol: str = None) -> bool:
    """
    Проверяет, не превышен ли лимит просадки.
    Возвращает True если торговля разрешена, False если нужно остановиться.
    """
    stats = journal.get_statistics(symbol=symbol)
    if stats['max_drawdown'] > MAX_DRAWDOWN * 100:  # max_drawdown в процентах
        logger.warning(
            f"⛔ Превышен лимит просадки! "
            f"Текущая: {stats['max_drawdown']:.2f}% > Лимит: {MAX_DRAWDOWN * 100:.1f}%"
        )
        return False
    return True


def calculate_position_size(balance: float, current_price: float,
                            atr: float = None, symbol: str = None) -> float:
    """Динамический размер позиции с Fractional Kelly и ATR"""
    # Проверка лимита просадки
    if not check_drawdown_limit(symbol):
        logger.warning(f"Торговля остановлена для {symbol}: превышен лимит просадки")
        return 0.0

    risk_amount = balance * RISK_PER_TRADE
    kelly_fraction = calculate_kelly(symbol)

    # Если Kelly = 0 (отрицательное матожидание) — не торгуем
    if kelly_fraction <= 0:
        logger.warning(f"Kelly <= 0 для {symbol}, пропускаем сделку")
        return 0.0

    risk_amount *= kelly_fraction

    if atr is None or atr <= 0:
        size = risk_amount / current_price
    else:
        stop_distance = atr * SL_ATR_MULTIPLIER
        size = risk_amount / stop_distance

    # Защита: не больше 25% баланса на одну позицию
    max_size = balance * 0.25 / current_price
    size = min(size, max_size)
    return max(size, 0.0)


async def set_tp_sl_trailing(symbol: str, side: str, entry_price: float, atr: float):
    """
    Устанавливает Stop-Loss, Take-Profit и Trailing Stop для позиции на Binance Futures.
    """
    try:
        sl_distance = atr * SL_ATR_MULTIPLIER
        tp_distance = atr * TP_ATR_MULTIPLIER

        if side.lower() == 'buy':
            sl_price = round(entry_price - sl_distance, 2)
            tp_price = round(entry_price + tp_distance, 2)
            close_side = 'sell'
        else:
            sl_price = round(entry_price + sl_distance, 2)
            tp_price = round(entry_price - tp_distance, 2)
            close_side = 'buy'

        # Получаем размер позиции
        positions = exchange.fetch_positions([symbol])
        amt = 0.0
        for p in positions:
            if float(p.get('contracts', 0)) != 0:
                amt = abs(float(p['contracts']))
                break

        if amt == 0:
            logger.warning(f"Позиция не найдена для {symbol}, TP/SL не установлены")
            return None

        common_params = {'reduceOnly': True, 'workingType': 'MARK_PRICE'}

        # Stop Loss
        exchange.create_order(symbol, 'STOP_MARKET', close_side, amt, None,
                              {**common_params, 'stopPrice': sl_price})

        # Take Profit
        result = exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', close_side, amt, None,
                                       {**common_params, 'stopPrice': tp_price})

        if USE_TRAILING and TRAILING_STOP_PCT > 0:
            exchange.create_order(symbol, 'TRAILING_STOP_MARKET', close_side, amt, None,
                                  {**common_params, 'callbackRate': TRAILING_STOP_PCT})

        logger.success(
            f"TP/SL{' + TrailingStop ' + str(TRAILING_STOP_PCT) + '%' if USE_TRAILING else ''} "
            f"установлены для {side.upper()} {symbol} | "
            f"SL: {sl_price} | TP: {tp_price} | ATR: {atr:.2f}"
        )
        return result

    except Exception as e:
        logger.error(f"Ошибка установки TP/SL/Trailing для {symbol}: {e}")
        logger.debug(f"Параметры: side={side}, entry={entry_price}, atr={atr}")
        raise
