"""
Журнал сделок: логирование, аналитика, расчёт статистики для Kelly.
Хранит историю в CSV для постфактум-анализа.
"""

import csv
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
from loguru import logger


import os as _os
_os.makedirs("data", exist_ok=True)
JOURNAL_FILE = "data/trade_journal.csv"
JOURNAL_FIELDS = [
    "timestamp", "symbol", "side", "entry_price", "exit_price",
    "amount", "pnl_pct", "pnl_abs", "exit_reason", "prediction", "actual_price"
]


@dataclass
class TradeRecord:
    timestamp: str
    symbol: str
    side: str  # 'buy' / 'sell'
    entry_price: float
    exit_price: Optional[float] = None
    amount: float = 0.0
    pnl_pct: Optional[float] = None
    pnl_abs: Optional[float] = None
    exit_reason: str = ""  # 'signal_change', 'stop_loss', 'take_profit', 'trailing'
    prediction: Optional[float] = None
    actual_price: Optional[float] = None


class TradeJournal:
    def __init__(self, filepath: str = JOURNAL_FILE):
        self.filepath = filepath
        self._ensure_file()
        self._open_trades: dict[str, TradeRecord] = {}  # symbol -> TradeRecord

    def _ensure_file(self):
        """Создаёт CSV файл с заголовками, если его нет"""
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
                writer.writeheader()
            logger.info(f"Создан журнал сделок: {self.filepath}")

    def open_trade(self, symbol: str, side: str, entry_price: float,
                   amount: float, prediction: float = None):
        """Регистрирует открытие сделки"""
        record = TradeRecord(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            amount=amount,
            prediction=prediction,
        )
        self._open_trades[symbol] = record
        logger.info(
            f"📝 Открыта сделка: {side.upper()} {symbol} @ {entry_price:.2f} | "
            f"Размер: {amount:.6f} | Предсказание: {prediction}"
        )

    def close_trade(self, symbol: str, exit_price: float, exit_reason: str = "signal_change"):
        """Закрывает сделку и записывает в журнал"""
        record = self._open_trades.pop(symbol, None)
        if record is None:
            logger.warning(f"Нет открытой сделки для {symbol}")
            return None

        record.exit_price = exit_price
        record.actual_price = exit_price
        record.exit_reason = exit_reason

        # PnL расчёт
        if record.side == 'buy':
            record.pnl_pct = (exit_price - record.entry_price) / record.entry_price * 100
        else:
            record.pnl_pct = (record.entry_price - exit_price) / record.entry_price * 100
        record.pnl_abs = record.pnl_pct / 100 * record.entry_price * record.amount

        # Запись в CSV
        self._write_record(record)
        logger.info(
            f"📝 Закрыта сделка: {symbol} | PnL: {record.pnl_pct:+.2f}% "
            f"({record.pnl_abs:+.4f} USDT) | Причина: {exit_reason}"
        )
        return record

    def _write_record(self, record: TradeRecord):
        """Записывает одну сделку в CSV"""
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
            writer.writerow(asdict(record))

    def get_statistics(self, symbol: str = None, last_n: int = None) -> dict:
        """
        Рассчитывает статистику по журналу для динамического Kelly.
        Возвращает: win_rate, avg_win, avg_loss, total_trades, total_pnl, max_drawdown
        """
        trades = self._load_trades(symbol)
        if last_n and len(trades) > last_n:
            trades = trades[-last_n:]

        if not trades:
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'total_trades': 0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
                'sharpe_like': 0.0,
            }

        wins = [t['pnl_pct'] for t in trades if t['pnl_pct'] and float(t['pnl_pct']) > 0]
        losses = [t['pnl_pct'] for t in trades if t['pnl_pct'] and float(t['pnl_pct']) <= 0]
        all_pnl = [float(t['pnl_pct']) for t in trades if t['pnl_pct']]

        win_rate = len(wins) / len(trades) if trades else 0.0
        avg_win = sum(float(w) for w in wins) / len(wins) / 100 if wins else 0.0
        avg_loss = abs(sum(float(l) for l in losses) / len(losses) / 100) if losses else 0.0

        # Max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in all_pnl:
            cumulative += pnl
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        # Sharpe-like ratio
        import numpy as np
        if len(all_pnl) > 1:
            arr = np.array(all_pnl)
            sharpe = arr.mean() / (arr.std() + 1e-10)
        else:
            sharpe = 0.0

        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(trades),
            'total_pnl': sum(all_pnl),
            'max_drawdown': max_dd,
            'sharpe_like': sharpe,
        }

    def _load_trades(self, symbol: str = None) -> list:
        """Загружает все завершённые сделки из CSV"""
        trades = []
        if not os.path.exists(self.filepath):
            return trades
        with open(self.filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if symbol and row['symbol'] != symbol:
                    continue
                if row.get('exit_price'):  # только закрытые сделки
                    trades.append(row)
        return trades

    def has_open_trade(self, symbol: str) -> bool:
        return symbol in self._open_trades

    def get_open_trade(self, symbol: str) -> Optional[TradeRecord]:
        return self._open_trades.get(symbol)


# Глобальный экземпляр
journal = TradeJournal()
