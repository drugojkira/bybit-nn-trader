"""
Trading Metrics v6 — метрики качества торговых сигналов.

Не просто accuracy, а метрики, важные для трейдинга:
  - Direction accuracy (общая + per-class)
  - Profitable signal rate (сигналы, приводящие к прибыли)
  - Expected return per signal
  - Sharpe ratio сигналов
  - Calibration error (ECE) — насколько confidence соответствует реальности
  - Agreement quality — корреляция agreement и accuracy
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Accuracy по направлениям.

    Args:
        y_true: true classes (0=down, 1=flat, 2=up)
        y_pred: predicted classes

    Returns:
        overall accuracy + per-class accuracy
    """
    assert len(y_true) == len(y_pred), "Length mismatch"

    overall = float(np.mean(y_true == y_pred))

    per_class = {}
    class_names = {0: 'down', 1: 'flat', 2: 'up'}
    for cls, name in class_names.items():
        mask = y_true == cls
        if mask.sum() > 0:
            per_class[name] = {
                'accuracy': float(np.mean(y_pred[mask] == cls)),
                'count': int(mask.sum()),
                'predicted_count': int((y_pred == cls).sum()),
            }
        else:
            per_class[name] = {'accuracy': 0.0, 'count': 0, 'predicted_count': 0}

    return {
        'overall_accuracy': overall,
        'per_class': per_class,
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Матрица ошибок."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_cls, pred_cls in zip(y_true, y_pred):
        cm[int(true_cls), int(pred_cls)] += 1
    return cm


def profitable_signal_rate(
    directions_pred: np.ndarray,
    returns_actual: np.ndarray,
    flat_threshold: float = 0.0001,
) -> Dict:
    """
    Доля сигналов, приводящих к прибыли.

    Args:
        directions_pred: предсказанные направления (-1, 0, +1)
        returns_actual: реальные returns за горизонт

    Returns:
        Метрики прибыльности сигналов.
    """
    # Только ненулевые сигналы (long/short)
    trade_mask = directions_pred != 0
    if trade_mask.sum() == 0:
        return {
            'profitable_rate': 0.0,
            'total_signals': 0,
            'profitable_signals': 0,
            'avg_return': 0.0,
        }

    trade_dirs = directions_pred[trade_mask]
    trade_returns = returns_actual[trade_mask]

    # Прибыль = direction * return > 0
    signal_returns = trade_dirs * trade_returns
    profitable = signal_returns > flat_threshold
    losing = signal_returns < -flat_threshold

    return {
        'profitable_rate': float(profitable.mean()),
        'losing_rate': float(losing.mean()),
        'total_signals': int(trade_mask.sum()),
        'profitable_signals': int(profitable.sum()),
        'losing_signals': int(losing.sum()),
        'avg_return': float(signal_returns.mean()),
        'median_return': float(np.median(signal_returns)),
        'avg_profit': float(signal_returns[profitable].mean()) if profitable.sum() > 0 else 0.0,
        'avg_loss': float(signal_returns[losing].mean()) if losing.sum() > 0 else 0.0,
    }


def signal_sharpe_ratio(
    directions_pred: np.ndarray,
    returns_actual: np.ndarray,
    annualization_factor: float = np.sqrt(365 * 24),  # для часовых свечей
) -> float:
    """
    Sharpe ratio торговых сигналов.

    signal_return = direction * actual_return
    Sharpe = mean(signal_returns) / std(signal_returns) * sqrt(periods/year)
    """
    trade_mask = directions_pred != 0
    if trade_mask.sum() < 10:
        return 0.0

    signal_returns = directions_pred[trade_mask] * returns_actual[trade_mask]

    std = np.std(signal_returns)
    if std < 1e-10:
        return 0.0

    return float((np.mean(signal_returns) / std) * annualization_factor)


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    """
    Expected Calibration Error (ECE).

    Если модель говорит confidence=0.8, то она должна быть права в 80% случаев.

    Args:
        confidences: массив confidence [0, 1]
        correct: массив bool (предсказание было правильным)

    Returns:
        ECE + детали по бинам
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)

        if mask.sum() == 0:
            continue

        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        bin_size = mask.sum()

        gap = abs(bin_acc - bin_conf)
        ece += gap * bin_size / len(confidences)

        bin_details.append({
            'range': f'{lo:.1f}-{hi:.1f}',
            'avg_confidence': float(bin_conf),
            'avg_accuracy': float(bin_acc),
            'count': int(bin_size),
            'gap': float(gap),
        })

    return {
        'ece': float(ece),
        'bins': bin_details,
    }


def agreement_quality(
    agreements: np.ndarray,
    correct: np.ndarray,
) -> Dict:
    """
    Насколько agreement между моделями коррелирует с правильностью.

    Хороший ансамбль: высокое agreement → высокая accuracy.
    """
    if len(agreements) < 10:
        return {'correlation': 0.0, 'high_agreement_accuracy': 0.0}

    # Корреляция
    corr = float(np.corrcoef(agreements, correct.astype(float))[0, 1])

    # Accuracy при высоком agreement (>0.7)
    high_mask = agreements > 0.7
    high_acc = float(correct[high_mask].mean()) if high_mask.sum() > 0 else 0.0

    # Accuracy при низком agreement (<0.5)
    low_mask = agreements < 0.5
    low_acc = float(correct[low_mask].mean()) if low_mask.sum() > 0 else 0.0

    return {
        'correlation': corr,
        'high_agreement_accuracy': high_acc,
        'low_agreement_accuracy': low_acc,
        'high_agreement_count': int(high_mask.sum()),
        'low_agreement_count': int(low_mask.sum()),
        'accuracy_lift': high_acc - low_acc,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    agreements: Optional[np.ndarray] = None,
    returns_actual: Optional[np.ndarray] = None,
) -> Dict:
    """
    Вычисляет все метрики разом.

    Args:
        y_true: true classes (0, 1, 2)
        y_pred: predicted classes (0, 1, 2)
        confidences: model confidence для каждого предсказания
        agreements: agreement scores (опц.)
        returns_actual: реальные returns (опц.)

    Returns:
        Полный набор метрик.
    """
    metrics = {}

    # Direction accuracy
    metrics['direction'] = direction_accuracy(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Calibration
    correct = (y_true == y_pred).astype(float)
    metrics['calibration'] = expected_calibration_error(confidences, correct)

    # Agreement quality
    if agreements is not None:
        metrics['agreement'] = agreement_quality(agreements, y_true == y_pred)

    # Trading metrics (need returns)
    if returns_actual is not None:
        # Map classes to directions
        cls_to_dir = {0: -1, 1: 0, 2: 1}
        dirs_pred = np.array([cls_to_dir[c] for c in y_pred])

        metrics['profitability'] = profitable_signal_rate(dirs_pred, returns_actual)
        metrics['sharpe'] = signal_sharpe_ratio(dirs_pred, returns_actual)

    return metrics


def format_metrics_report(metrics: Dict) -> str:
    """Форматирует метрики в читаемый отчёт."""
    lines = []
    lines.append("=" * 50)
    lines.append("MODEL EVALUATION REPORT")
    lines.append("=" * 50)

    if 'direction' in metrics:
        d = metrics['direction']
        lines.append(f"\nOverall Accuracy: {d['overall_accuracy']:.4f}")
        lines.append("Per-class:")
        for cls, info in d['per_class'].items():
            lines.append(f"  {cls}: acc={info['accuracy']:.3f} (n={info['count']}, pred={info['predicted_count']})")

    if 'calibration' in metrics:
        lines.append(f"\nCalibration ECE: {metrics['calibration']['ece']:.4f}")

    if 'agreement' in metrics:
        a = metrics['agreement']
        lines.append(f"\nAgreement-Accuracy Correlation: {a['correlation']:.3f}")
        lines.append(f"  High agreement acc: {a['high_agreement_accuracy']:.3f} (n={a['high_agreement_count']})")
        lines.append(f"  Low agreement acc:  {a['low_agreement_accuracy']:.3f} (n={a['low_agreement_count']})")
        lines.append(f"  Accuracy lift:      {a['accuracy_lift']:.3f}")

    if 'profitability' in metrics:
        p = metrics['profitability']
        lines.append(f"\nProfitable Rate: {p['profitable_rate']:.3f} ({p['profitable_signals']}/{p['total_signals']})")
        lines.append(f"Avg Return/Signal: {p['avg_return']:.6f}")
        lines.append(f"Avg Profit: {p['avg_profit']:.6f}")
        lines.append(f"Avg Loss:   {p['avg_loss']:.6f}")

    if 'sharpe' in metrics:
        lines.append(f"\nSignal Sharpe Ratio: {metrics['sharpe']:.3f}")

    lines.append("=" * 50)
    return "\n".join(lines)
