"""
Walk-Forward Training Pipeline v6.

Стратегия валидации:
  - Walk-Forward CV с embargo-периодом (нет утечки данных между фолдами)
  - Расширяющееся окно обучения (expanding window)
  - Независимое обучение TFT, LightGBM, TCN
  - Калибровка Meta-Learner на OOS-данных
"""

import numpy as np
import pandas as pd
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger

from data.feature_engine import build_features, compute_targets
from data.normalizer import TrainValNormalizer
from models.tft_model import TFTWrapper
from models.lgbm_model import LGBMWrapper
from models.tcn_model import TCNWrapper
from models.meta_learner import MetaLearner
from models.regime_detector import RegimeDetector


class WalkForwardSplit:
    """
    Генератор Walk-Forward фолдов с embargo.

    Схема одного фолда:
        [======= TRAIN =======][EMBARGO][=== VAL ===]

    Expanding window: каждый следующий фолд расширяет train.
    """

    def __init__(
        self,
        n_samples: int,
        n_splits: int = 5,
        val_ratio: float = 0.15,
        embargo_bars: int = 12,
        min_train_bars: int = 200,
    ):
        self.n_samples = n_samples
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.embargo_bars = embargo_bars
        self.min_train_bars = min_train_bars

    def split(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Возвращает список (train_idx, val_idx) для каждого фолда.
        """
        val_size = max(int(self.n_samples * self.val_ratio), 50)
        splits = []

        for i in range(self.n_splits):
            # Конец val двигается от конца к началу
            val_end = self.n_samples - i * (val_size // 2)
            val_start = val_end - val_size

            # Embargo
            train_end = val_start - self.embargo_bars

            if train_end < self.min_train_bars:
                logger.warning(f"Fold {i}: недостаточно данных для train ({train_end} < {self.min_train_bars})")
                continue

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)

            splits.append((train_idx, val_idx))
            logger.info(
                f"Fold {i}: train[0:{train_end}] ({len(train_idx)}), "
                f"embargo[{train_end}:{val_start}] ({self.embargo_bars}), "
                f"val[{val_start}:{val_end}] ({len(val_idx)})"
            )

        return list(reversed(splits))  # от самого раннего к позднему


class TrainPipeline:
    """
    Полный пайплайн обучения ансамбля.

    Этапы:
      1. Генерация фичей и таргетов
      2. Walk-Forward CV
      3. Обучение каждой модели
      4. Калибровка Meta-Learner на OOS-предсказаниях
      5. Обучение Regime Detector
      6. Сохранение всех артефактов
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Walk-Forward params
        self.n_splits = self.config.get('n_splits', 5)
        self.val_ratio = self.config.get('val_ratio', 0.15)
        self.embargo_bars = self.config.get('embargo_bars', 12)

        # Model configs
        self.tft_config = self.config.get('tft', {})
        self.lgbm_config = self.config.get('lgbm', {})
        self.tcn_config = self.config.get('tcn', {})
        self.regime_config = self.config.get('regime', {})

        # Training params
        self.lookback = self.config.get('lookback', 60)
        self.target_horizon = self.config.get('target_horizon', 6)
        self.direction_threshold = self.config.get('direction_threshold', 0.001)

        # Results
        self.fold_results: List[Dict] = []
        self.best_models = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Вычисляет фичи и таргеты из сырых OHLCV.

        Args:
            df: DataFrame с колонками [open, high, low, close, volume, timestamp]
            multi_tf_data: {'15m': df_15m, '1h': df_1h, '4h': df_4h}

        Returns:
            features_df: DataFrame с фичами
            targets: массив направлений (0=down, 1=flat, 2=up)
        """
        logger.info(f"Подготовка данных: {len(df)} свечей")

        # Фичи
        features_df = build_features(df, multi_tf_data)

        # Таргеты
        targets_df = compute_targets(
            df,
            horizons=[self.target_horizon],
            direction_threshold=self.direction_threshold,
        )

        target_col = f'direction_{self.target_horizon}'
        if target_col not in targets_df.columns:
            raise ValueError(f"Target column '{target_col}' not found. Available: {targets_df.columns.tolist()}")

        # Align
        common_idx = features_df.index.intersection(targets_df.index)
        features_df = features_df.loc[common_idx]
        targets = targets_df.loc[common_idx, target_col].values

        # Map direction: -1 → 0, 0 → 1, 1 → 2
        targets_mapped = np.zeros_like(targets, dtype=np.int64)
        targets_mapped[targets == -1] = 0
        targets_mapped[targets == 0] = 1
        targets_mapped[targets == 1] = 2

        # Drop NaN rows
        valid_mask = ~(features_df.isna().any(axis=1) | np.isnan(targets_mapped.astype(float)))
        features_df = features_df[valid_mask]
        targets_mapped = targets_mapped[valid_mask]

        logger.info(
            f"Данные готовы: {len(features_df)} сэмплов, "
            f"{features_df.shape[1]} фичей, "
            f"классы: {dict(zip(*np.unique(targets_mapped, return_counts=True)))}"
        )

        return features_df, targets_mapped

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Создаёт 3D-тензоры (N, lookback, features) для TFT/TCN."""
        sequences = []
        labels = []
        for i in range(self.lookback, len(X)):
            sequences.append(X[i - self.lookback:i])
            labels.append(y[i])
        return np.array(sequences), np.array(labels)

    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        fold_idx: int,
    ) -> Dict:
        """
        Обучает все 3 модели на одном фолде.

        Returns:
            Результаты валидации каждой модели.
        """
        results = {'fold': fold_idx}
        logger.info(f"=== Fold {fold_idx}: train={len(X_train)}, val={len(X_val)} ===")

        # --- Normalize ---
        normalizer = TrainValNormalizer()
        X_train_norm = normalizer.fit_transform(X_train)
        X_val_norm = normalizer.transform(X_val)

        # --- Sequences for TFT/TCN ---
        X_train_seq, y_train_seq = self._create_sequences(X_train_norm, y_train)
        X_val_seq, y_val_seq = self._create_sequences(X_val_norm, y_val)

        if len(X_train_seq) < 50 or len(X_val_seq) < 10:
            logger.warning(f"Fold {fold_idx}: недостаточно сэмплов после создания последовательностей")
            return results

        # --- 1. TFT ---
        try:
            logger.info(f"Fold {fold_idx}: Training TFT...")
            tft = TFTWrapper(self.tft_config)
            tft_metrics = tft.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            results['tft'] = tft_metrics
            results['tft_model'] = tft
            logger.info(f"Fold {fold_idx} TFT: {tft_metrics}")
        except Exception as e:
            logger.error(f"Fold {fold_idx} TFT failed: {e}")
            results['tft'] = {'error': str(e)}

        # --- 2. LightGBM (flat features — последний таймстеп) ---
        try:
            logger.info(f"Fold {fold_idx}: Training LightGBM...")
            lgbm = LGBMWrapper(self.lgbm_config)

            # LightGBM получает плоский вектор: последний таймстеп из sequence
            X_train_flat = X_train_seq[:, -1, :]  # (N, features)
            X_val_flat = X_val_seq[:, -1, :]

            lgbm.feature_names = feature_names
            lgbm_metrics = lgbm.train(X_train_flat, y_train_seq, X_val_flat, y_val_seq)
            results['lgbm'] = lgbm_metrics
            results['lgbm_model'] = lgbm
            logger.info(f"Fold {fold_idx} LGBM: {lgbm_metrics}")
        except Exception as e:
            logger.error(f"Fold {fold_idx} LGBM failed: {e}")
            results['lgbm'] = {'error': str(e)}

        # --- 3. TCN ---
        try:
            logger.info(f"Fold {fold_idx}: Training TCN...")
            tcn = TCNWrapper(self.tcn_config)
            tcn_metrics = tcn.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            results['tcn'] = tcn_metrics
            results['tcn_model'] = tcn
            logger.info(f"Fold {fold_idx} TCN: {tcn_metrics}")
        except Exception as e:
            logger.error(f"Fold {fold_idx} TCN failed: {e}")
            results['tcn'] = {'error': str(e)}

        return results

    def calibrate_meta_learner(
        self,
        fold_results: List[Dict],
        X_val_all: np.ndarray,
        y_val_all: np.ndarray,
    ) -> MetaLearner:
        """
        Калибрует Meta-Learner на OOS-предсказаниях последнего фолда.
        """
        meta = MetaLearner(
            window=self.config.get('meta_window', 100),
            temperature=self.config.get('meta_temperature', 2.0),
        )

        # Берём модели из последнего фолда
        last_fold = fold_results[-1]

        # Создаём sequences
        X_val_seq, y_val_seq = self._create_sequences(X_val_all, y_val_all)
        if len(X_val_seq) == 0:
            logger.warning("Нет данных для калибровки Meta-Learner")
            return meta

        # Собираем предсказания каждой модели
        for i in range(len(X_val_seq)):
            predictions = {}
            sample = X_val_seq[i:i+1]
            flat_sample = sample[:, -1, :]

            for model_name in ['tft', 'tcn']:
                model = last_fold.get(f'{model_name}_model')
                if model and model.is_trained:
                    try:
                        pred = model.predict(sample[0])
                        predictions[model_name] = pred
                    except Exception:
                        pass

            lgbm_model = last_fold.get('lgbm_model')
            if lgbm_model and lgbm_model.is_trained:
                try:
                    pred = lgbm_model.predict(flat_sample[0])
                    predictions['lgbm'] = pred
                except Exception:
                    pass

            # Маппим actual direction
            actual_cls = y_val_seq[i]
            actual_dir = {0: -1, 1: 0, 2: 1}.get(actual_cls, 0)

            meta.record_all_outcomes(actual_dir, predictions)

        logger.info(f"Meta-Learner calibrated: {meta.get_stats()}")
        return meta

    def train_regime_detector(self, df: pd.DataFrame) -> RegimeDetector:
        """Обучает HMM Regime Detector на всех данных."""
        logger.info("Training Regime Detector...")
        detector = RegimeDetector(self.regime_config)
        detector.fit(df)
        return detector

    def run(
        self,
        df: pd.DataFrame,
        multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
        save_path: Optional[str] = None,
    ) -> Dict:
        """
        Полный запуск пайплайна.

        Args:
            df: OHLCV DataFrame
            multi_tf_data: мультитаймфреймовые данные
            save_path: путь для сохранения моделей

        Returns:
            Результаты обучения (метрики по фолдам, финальные веса).
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("TRAIN PIPELINE START")
        logger.info("=" * 60)

        # 1. Prepare data
        features_df, targets = self.prepare_data(df, multi_tf_data)
        feature_names = features_df.columns.tolist()
        X = features_df.values

        # 2. Walk-Forward splits
        wf = WalkForwardSplit(
            n_samples=len(X),
            n_splits=self.n_splits,
            val_ratio=self.val_ratio,
            embargo_bars=self.embargo_bars,
        )
        splits = wf.split()

        if not splits:
            raise ValueError("Нет валидных фолдов. Нужно больше данных.")

        # 3. Train each fold
        self.fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train, y_train = X[train_idx], targets[train_idx]
            X_val, y_val = X[val_idx], targets[val_idx]

            fold_result = self.train_fold(
                X_train, y_train, X_val, y_val,
                feature_names, fold_idx,
            )
            self.fold_results.append(fold_result)

        # 4. Calibrate Meta-Learner on last fold's validation set
        last_train_idx, last_val_idx = splits[-1]
        normalizer = TrainValNormalizer()
        normalizer.fit_transform(X[last_train_idx])
        X_val_norm = normalizer.transform(X[last_val_idx])

        meta_learner = self.calibrate_meta_learner(
            self.fold_results, X_val_norm, targets[last_val_idx]
        )

        # 5. Train Regime Detector on full data
        regime_detector = self.train_regime_detector(df)

        # 6. Final results
        elapsed = time.time() - start_time
        summary = self._summarize_results()
        summary['elapsed_seconds'] = elapsed
        summary['meta_learner_stats'] = meta_learner.get_stats()

        logger.info("=" * 60)
        logger.info(f"TRAIN PIPELINE COMPLETE ({elapsed:.1f}s)")
        logger.info(f"Summary: {summary}")
        logger.info("=" * 60)

        # 7. Save models
        if save_path:
            self._save_all(save_path, self.fold_results[-1], meta_learner, regime_detector, normalizer)

        return {
            'summary': summary,
            'fold_results': self.fold_results,
            'meta_learner': meta_learner,
            'regime_detector': regime_detector,
            'normalizer': normalizer,
        }

    def _summarize_results(self) -> Dict:
        """Средние метрики по фолдам."""
        summary = {}
        for model_name in ['tft', 'lgbm', 'tcn']:
            accs = []
            for fr in self.fold_results:
                if model_name in fr and 'accuracy' in fr.get(model_name, {}):
                    accs.append(fr[model_name]['accuracy'])
                elif model_name in fr and 'val_accuracy' in fr.get(model_name, {}):
                    accs.append(fr[model_name]['val_accuracy'])
            if accs:
                summary[f'{model_name}_mean_acc'] = float(np.mean(accs))
                summary[f'{model_name}_std_acc'] = float(np.std(accs))
        return summary

    def _save_all(
        self,
        save_path: str,
        last_fold: Dict,
        meta_learner: MetaLearner,
        regime_detector: RegimeDetector,
        normalizer: TrainValNormalizer,
    ):
        """Сохраняет все модели и артефакты."""
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        # Save models from last fold
        for model_name in ['tft', 'lgbm', 'tcn']:
            model = last_fold.get(f'{model_name}_model')
            if model and model.is_trained:
                try:
                    model.save(str(path / model_name))
                    logger.info(f"Saved {model_name} to {path / model_name}")
                except Exception as e:
                    logger.error(f"Failed to save {model_name}: {e}")

        # Save regime detector
        try:
            regime_detector.save(str(path / 'regime'))
            logger.info(f"Saved regime detector to {path / 'regime'}")
        except Exception as e:
            logger.error(f"Failed to save regime detector: {e}")

        # Save normalizer
        try:
            normalizer.save(str(path / 'normalizer.json'))
            logger.info(f"Saved normalizer to {path / 'normalizer.json'}")
        except Exception as e:
            logger.error(f"Failed to save normalizer: {e}")

        logger.info(f"All models saved to {path}")


class QuickRetrain:
    """
    Быстрое дообучение на свежих данных (каждые 6ч).
    Не пересоздаёт модели — только fine-tune на последних N свечах.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.retrain_bars = self.config.get('retrain_bars', 500)
        self.lookback = self.config.get('lookback', 60)

    def retrain(
        self,
        df_recent: pd.DataFrame,
        tft: TFTWrapper,
        lgbm: LGBMWrapper,
        tcn: TCNWrapper,
        meta_learner: MetaLearner,
        normalizer: TrainValNormalizer,
    ) -> Dict:
        """
        Fine-tune моделей на последних данных.
        """
        logger.info(f"Quick retrain на {len(df_recent)} свечах")

        features_df = build_features(df_recent)
        targets_df = compute_targets(df_recent, horizons=[6], direction_threshold=0.001)

        target_col = 'direction_6'
        common_idx = features_df.index.intersection(targets_df.index)
        features_df = features_df.loc[common_idx]
        targets = targets_df.loc[common_idx, target_col].values

        # Map targets
        y = np.zeros_like(targets, dtype=np.int64)
        y[targets == -1] = 0
        y[targets == 0] = 1
        y[targets == 1] = 2

        # Drop NaN
        valid = ~(features_df.isna().any(axis=1))
        X = features_df[valid].values
        y = y[valid.values]

        # Normalize
        X_norm = normalizer.transform(X)

        # Train/val split (last 20% as val)
        split_idx = int(len(X_norm) * 0.8)
        X_train, X_val = X_norm[:split_idx], X_norm[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Sequences
        def make_seq(data, labels):
            seqs, lbls = [], []
            for i in range(self.lookback, len(data)):
                seqs.append(data[i - self.lookback:i])
                lbls.append(labels[i])
            return np.array(seqs), np.array(lbls)

        X_train_seq, y_train_seq = make_seq(X_train, y_train)
        X_val_seq, y_val_seq = make_seq(X_val, y_val)

        results = {}

        if len(X_train_seq) < 20:
            logger.warning("Недостаточно данных для quick retrain")
            return results

        # Fine-tune TFT (меньше эпох)
        if tft and tft.is_trained:
            try:
                r = tft.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, max_epochs=10)
                results['tft'] = r
                logger.info(f"TFT retrained: {r}")
            except Exception as e:
                logger.error(f"TFT retrain failed: {e}")

        # Fine-tune TCN
        if tcn and tcn.is_trained:
            try:
                r = tcn.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, max_epochs=10)
                results['tcn'] = r
                logger.info(f"TCN retrained: {r}")
            except Exception as e:
                logger.error(f"TCN retrain failed: {e}")

        # LGBM — полное переобучение (быстрое)
        if lgbm:
            try:
                X_train_flat = X_train_seq[:, -1, :]
                X_val_flat = X_val_seq[:, -1, :]
                r = lgbm.train(X_train_flat, y_train_seq, X_val_flat, y_val_seq)
                results['lgbm'] = r
                logger.info(f"LGBM retrained: {r}")
            except Exception as e:
                logger.error(f"LGBM retrain failed: {e}")

        return results
