"""
Trade Decision Engine v6 — финальное решение о входе в сделку.
Учитывает: сигнал ансамбля + режим рынка + риск-менеджмент + уверенность.
"""

from loguru import logger
from typing import Dict, Optional


class TradeDecisionEngine:
    """
    Принимает финальное решение о торговле на основе:
    1. Сигнала meta-learner (направление + уверенность + согласие)
    2. Параметров текущего режима рынка
    3. Состояния портфеля (открытые позиции, drawdown)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.min_agreement = self.config.get('min_agreement', 0.5)
        self.min_risk_reward = self.config.get('min_risk_reward', 1.5)
        self.counter_trend_scale = self.config.get('counter_trend_scale', 0.5)

    def should_trade(
        self,
        meta_signal: Dict,
        regime: str,
        regime_params: Dict,
        portfolio_state: Optional[Dict] = None,
    ) -> Dict:
        """
        Решает: торговать или нет.

        Args:
            meta_signal: результат MetaLearner.combine_predictions()
            regime: текущий режим ('trending_up', 'trending_down', 'ranging', 'volatile')
            regime_params: параметры режима из RegimeDetector
            portfolio_state: {'open_positions': int, 'max_positions': int, 'drawdown': float}

        Returns:
            {
                'action': 'long' / 'short' / 'hold',
                'size_multiplier': float (0.0 - 1.5),
                'confidence': float,
                'regime': str,
                'reasons': list[str],  # почему принято это решение
            }
        """
        reasons = []

        # 1. Режим запрещает торговлю?
        if not regime_params.get('trade', True):
            reasons.append(f'regime={regime} (торговля запрещена)')
            return self._hold(reasons, regime)

        # 2. Direction = flat?
        direction = meta_signal.get('direction', 0)
        if direction == 0:
            reasons.append('signal=flat')
            return self._hold(reasons, regime)

        # 3. Уверенность ниже порога режима?
        confidence = meta_signal.get('confidence', 0.0)
        min_conf = regime_params.get('min_confidence', 0.6)
        if confidence < min_conf:
            reasons.append(f'confidence={confidence:.2f} < {min_conf:.2f}')
            return self._hold(reasons, regime)

        # 4. Модели не согласны?
        agreement = meta_signal.get('agreement', 0.0)
        if agreement < self.min_agreement:
            reasons.append(f'agreement={agreement:.2f} < {self.min_agreement:.2f}')
            return self._hold(reasons, regime)

        # 5. Портфель: лимит позиций?
        if portfolio_state:
            open_pos = portfolio_state.get('open_positions', 0)
            max_pos = portfolio_state.get('max_positions', 5)
            if open_pos >= max_pos:
                reasons.append(f'positions={open_pos}/{max_pos}')
                return self._hold(reasons, regime)

            # Drawdown limit
            dd = portfolio_state.get('drawdown', 0.0)
            max_dd = portfolio_state.get('max_drawdown', 10.0)
            if dd > max_dd:
                reasons.append(f'drawdown={dd:.1f}% > {max_dd:.1f}%')
                return self._hold(reasons, regime)

        # 6. Определяем действие
        action = 'long' if direction == 1 else 'short'

        # 7. Размер позиции: base * regime_scale * confidence
        size_mult = regime_params.get('position_scale', 1.0)

        # Контр-трендовая сделка?
        preferred = regime_params.get('prefer_direction')
        if preferred and preferred not in ('both', None):
            if action != preferred:
                size_mult *= self.counter_trend_scale
                reasons.append(f'counter-trend (x{self.counter_trend_scale})')

        # Масштабируем по уверенности
        size_mult *= confidence
        reasons.append(f'signal={action} conf={confidence:.2f} regime={regime}')

        return {
            'action': action,
            'size_multiplier': round(size_mult, 3),
            'confidence': confidence,
            'regime': regime,
            'sl_multiplier': regime_params.get('sl_multiplier', 2.0),
            'tp_multiplier': regime_params.get('tp_multiplier', 4.0),
            'reasons': reasons,
        }

    def _hold(self, reasons: list, regime: str) -> Dict:
        return {
            'action': 'hold',
            'size_multiplier': 0.0,
            'confidence': 0.0,
            'regime': regime,
            'reasons': reasons,
        }
