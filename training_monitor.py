"""
Модуль мониторинга обучения: сбор метрик и генерация графиков.
Данные хранятся в памяти и периодически сохраняются на диск.
"""

import os
import json
import io
import time
from datetime import datetime
from collections import defaultdict
from loguru import logger

try:
    import matplotlib
    matplotlib.use('Agg')  # Неинтерактивный бэкенд
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib не установлен — графики обучения недоступны")


MONITOR_DIR = "data/training_logs"
os.makedirs(MONITOR_DIR, exist_ok=True)


class TrainingMonitor:
    """Собирает метрики обучения и генерирует графики"""

    def __init__(self):
        # Метрики по символам: symbol -> list of dicts
        self.metrics: dict[str, list[dict]] = defaultdict(list)
        # Prediction accuracy tracking
        self.predictions: dict[str, list[dict]] = defaultdict(list)
        # Загружаем историю с диска если есть
        self._load_history()

    def _load_history(self):
        """Загружает историю метрик с диска"""
        for filename in os.listdir(MONITOR_DIR):
            if filename.startswith("metrics_") and filename.endswith(".json"):
                symbol = filename.replace("metrics_", "").replace(".json", "").replace("_", "/")
                filepath = os.path.join(MONITOR_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    self.metrics[symbol] = data[-2000:]  # Последние 2000 записей
                except Exception:
                    pass

    def save_history(self):
        """Сохраняет метрики на диск"""
        for symbol, data in self.metrics.items():
            safe_name = symbol.replace('/', '_').replace(':', '_')
            filepath = os.path.join(MONITOR_DIR, f"metrics_{safe_name}.json")
            with open(filepath, 'w') as f:
                json.dump(data[-2000:], f)

    def record_training(self, symbol: str, train_loss: float, val_loss: float,
                        epoch: int = 0, learning_rate: float = None):
        """Записывает результат шага обучения"""
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': learning_rate,
            'overfitting_ratio': val_loss / (train_loss + 1e-10),
        }
        self.metrics[symbol].append(record)

        # Ограничиваем размер в памяти
        if len(self.metrics[symbol]) > 2000:
            self.metrics[symbol] = self.metrics[symbol][-2000:]

        # Периодическое сохранение (каждые 50 записей)
        if len(self.metrics[symbol]) % 50 == 0:
            self.save_history()

    def record_prediction(self, symbol: str, predicted: float, actual: float,
                          signal: int, confidence: float = None):
        """Записывает предсказание для мониторинга точности"""
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'predicted': predicted,
            'actual': actual,
            'signal': signal,
            'confidence': confidence,
            'error': abs(predicted - actual),
            'error_pct': abs(predicted - actual) / (actual + 1e-10) * 100,
            'direction_correct': (predicted > actual) == (signal > 0) if signal != 0 else None,
        }
        self.predictions[symbol].append(record)
        if len(self.predictions[symbol]) > 1000:
            self.predictions[symbol] = self.predictions[symbol][-1000:]

    def get_training_summary(self, symbol: str) -> dict:
        """Краткая сводка обучения"""
        data = self.metrics.get(symbol, [])
        if not data:
            return {'status': 'no_data', 'total_steps': 0}

        recent = data[-50:]  # Последние 50 шагов
        all_train = [d['train_loss'] for d in recent if d['train_loss'] > 0]
        all_val = [d['val_loss'] for d in recent if d['val_loss'] > 0]

        return {
            'status': 'active',
            'total_steps': len(data),
            'avg_train_loss': sum(all_train) / len(all_train) if all_train else 0,
            'avg_val_loss': sum(all_val) / len(all_val) if all_val else 0,
            'min_val_loss': min(all_val) if all_val else 0,
            'last_train_loss': data[-1]['train_loss'],
            'last_val_loss': data[-1]['val_loss'],
            'overfitting_trend': self._check_overfitting_trend(data),
            'loss_trend': self._check_loss_trend(data),
        }

    def get_prediction_summary(self, symbol: str) -> dict:
        """Сводка точности предсказаний"""
        preds = self.predictions.get(symbol, [])
        if len(preds) < 10:
            return {'status': 'insufficient_data', 'count': len(preds)}

        recent = preds[-100:]
        errors_pct = [p['error_pct'] for p in recent]
        direction_checks = [p['direction_correct'] for p in recent if p['direction_correct'] is not None]

        return {
            'status': 'active',
            'count': len(preds),
            'recent_count': len(recent),
            'avg_error_pct': sum(errors_pct) / len(errors_pct),
            'max_error_pct': max(errors_pct),
            'direction_accuracy': sum(direction_checks) / len(direction_checks) if direction_checks else 0,
        }

    def _check_overfitting_trend(self, data: list[dict]) -> str:
        """Определяет тренд overfitting"""
        if len(data) < 20:
            return 'unknown'
        recent_ratios = [d['overfitting_ratio'] for d in data[-20:] if d['train_loss'] > 0]
        older_ratios = [d['overfitting_ratio'] for d in data[-40:-20] if d['train_loss'] > 0]
        if not recent_ratios or not older_ratios:
            return 'unknown'
        avg_recent = sum(recent_ratios) / len(recent_ratios)
        avg_older = sum(older_ratios) / len(older_ratios)
        if avg_recent > avg_older * 1.3:
            return 'worsening'
        elif avg_recent < avg_older * 0.8:
            return 'improving'
        return 'stable'

    def _check_loss_trend(self, data: list[dict]) -> str:
        """Определяет тренд loss"""
        if len(data) < 20:
            return 'unknown'
        recent = [d['val_loss'] for d in data[-10:] if d['val_loss'] > 0]
        older = [d['val_loss'] for d in data[-20:-10] if d['val_loss'] > 0]
        if not recent or not older:
            return 'unknown'
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        if avg_recent < avg_older * 0.95:
            return 'decreasing'
        elif avg_recent > avg_older * 1.05:
            return 'increasing'
        return 'plateau'

    # === Генерация графиков ===

    def generate_loss_chart(self, symbol: str, last_n: int = 200) -> io.BytesIO | None:
        """Генерирует график train/val loss"""
        if not HAS_MATPLOTLIB:
            return None

        data = self.metrics.get(symbol, [])
        if len(data) < 5:
            return None

        data = data[-last_n:]
        steps = list(range(len(data)))
        train_losses = [d['train_loss'] for d in data]
        val_losses = [d['val_loss'] for d in data]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f'Training Monitor — {symbol}', fontsize=14, fontweight='bold')

        # Верхний график: Loss
        ax1.plot(steps, train_losses, label='Train Loss', color='#2196F3', linewidth=1.5, alpha=0.8)
        ax1.plot(steps, val_losses, label='Val Loss', color='#FF5722', linewidth=1.5, alpha=0.8)
        ax1.fill_between(steps, train_losses, val_losses, alpha=0.1, color='red',
                         where=[v > t for v, t in zip(val_losses, train_losses)])
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_xlabel('Training Steps')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Train vs Validation Loss')

        # Нижний график: Overfitting ratio
        ratios = [d['overfitting_ratio'] for d in data]
        colors = ['#4CAF50' if r < 1.5 else '#FFC107' if r < 3 else '#F44336' for r in ratios]
        ax2.bar(steps, ratios, color=colors, alpha=0.7, width=1.0)
        ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal (1.0)')
        ax2.axhline(y=3.0, color='red', linestyle='--', alpha=0.5, label='Danger (3.0)')
        ax2.set_ylabel('Val/Train Ratio')
        ax2.set_xlabel('Training Steps')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Overfitting Indicator')

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf

    def generate_prediction_chart(self, symbol: str, last_n: int = 100) -> io.BytesIO | None:
        """Генерирует график предсказаний vs реальных цен"""
        if not HAS_MATPLOTLIB:
            return None

        preds = self.predictions.get(symbol, [])
        if len(preds) < 10:
            return None

        preds = preds[-last_n:]
        steps = list(range(len(preds)))
        predicted = [p['predicted'] for p in preds]
        actual = [p['actual'] for p in preds]
        errors = [p['error_pct'] for p in preds]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f'Prediction Accuracy — {symbol}', fontsize=14, fontweight='bold')

        # Верхний: Predicted vs Actual
        ax1.plot(steps, actual, label='Actual Price', color='#2196F3', linewidth=2)
        ax1.plot(steps, predicted, label='Predicted Price', color='#FF9800', linewidth=1.5,
                 linestyle='--', alpha=0.8)
        ax1.fill_between(steps, actual, predicted, alpha=0.15, color='orange')
        ax1.set_ylabel('Price (USDT)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Predicted vs Actual Price')

        # Нижний: Error %
        colors = ['#4CAF50' if e < 0.5 else '#FFC107' if e < 1.0 else '#F44336' for e in errors]
        ax2.bar(steps, errors, color=colors, alpha=0.7, width=1.0)
        ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Error %')
        ax2.set_xlabel('Candles')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Prediction Error (%)')

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf

    def generate_full_dashboard(self, symbol: str) -> io.BytesIO | None:
        """Полный дашборд: loss + predictions + statistics"""
        if not HAS_MATPLOTLIB:
            return None

        data = self.metrics.get(symbol, [])
        preds = self.predictions.get(symbol, [])

        if len(data) < 5 and len(preds) < 5:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Full Training Dashboard — {symbol}', fontsize=16, fontweight='bold')

        # 1. Loss chart (top-left)
        if data:
            recent_data = data[-200:]
            steps = list(range(len(recent_data)))
            axes[0, 0].plot(steps, [d['train_loss'] for d in recent_data],
                           label='Train', color='#2196F3', linewidth=1.5)
            axes[0, 0].plot(steps, [d['val_loss'] for d in recent_data],
                           label='Val', color='#FF5722', linewidth=1.5)
            axes[0, 0].legend()
            axes[0, 0].set_title('Loss History')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No training data', ha='center', va='center')

        # 2. Overfitting ratio (top-right)
        if data:
            recent_data = data[-100:]
            ratios = [d['overfitting_ratio'] for d in recent_data]
            colors = ['#4CAF50' if r < 1.5 else '#FFC107' if r < 3 else '#F44336' for r in ratios]
            axes[0, 1].bar(range(len(ratios)), ratios, color=colors, alpha=0.7)
            axes[0, 1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
            axes[0, 1].axhline(y=3.0, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Overfitting Ratio (val/train)')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No data', ha='center', va='center')

        # 3. Predictions vs actual (bottom-left)
        if len(preds) >= 10:
            recent_preds = preds[-100:]
            steps = list(range(len(recent_preds)))
            axes[1, 0].plot(steps, [p['actual'] for p in recent_preds],
                           label='Actual', color='#2196F3', linewidth=2)
            axes[1, 0].plot(steps, [p['predicted'] for p in recent_preds],
                           label='Predicted', color='#FF9800', linewidth=1.5, linestyle='--')
            axes[1, 0].legend()
            axes[1, 0].set_title('Price Predictions')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient predictions', ha='center', va='center')

        # 4. Statistics summary (bottom-right)
        ax = axes[1, 1]
        ax.axis('off')
        summary = self.get_training_summary(symbol)
        pred_summary = self.get_prediction_summary(symbol)

        stats_text = (
            f"Training Steps: {summary['total_steps']}\n"
            f"Last Train Loss: {summary.get('last_train_loss', 'N/A'):.6f}\n"
            f"Last Val Loss: {summary.get('last_val_loss', 'N/A'):.6f}\n"
            f"Min Val Loss: {summary.get('min_val_loss', 'N/A'):.6f}\n"
            f"Loss Trend: {summary.get('loss_trend', 'N/A')}\n"
            f"Overfitting: {summary.get('overfitting_trend', 'N/A')}\n"
            f"\n"
            f"Predictions: {pred_summary.get('count', 0)}\n"
            f"Direction Acc: {pred_summary.get('direction_accuracy', 0):.1%}\n"
            f"Avg Error: {pred_summary.get('avg_error_pct', 0):.3f}%\n"
        )
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Summary Statistics')

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf


# Глобальный экземпляр
monitor = TrainingMonitor()
