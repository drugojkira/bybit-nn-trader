"""
NN Trader v6 — Ensemble Trading System.

Архитектура:
  Data Pipeline → Regime Detector → [TFT + LightGBM + TCN] → Meta-Learner → Decision Engine → Trade

Ключевые отличия от v5:
  - 3 модели вместо одной LSTM
  - HMM Regime Detection для адаптации к рынку
  - Walk-Forward обучение без data leakage
  - Адаптивный Meta-Learner с весами по accuracy
  - Model Registry с версионированием и rollback
  - 50+ фичей вместо 12
"""

import asyncio
import os
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI
from loguru import logger
from datetime import datetime
from pathlib import Path

from config import (
    CCXT_SYMBOLS, TIMEFRAME, LOG_FILE, LOG_ROTATION, LOG_RETENTION,
    MAX_TOTAL_POSITIONS, REPORT_INTERVAL_CANDLES, AUTO_DASHBOARD_INTERVAL,
    # v6 config
    V6_LOOKBACK, V6_TARGET_HORIZON, V6_DIRECTION_THRESHOLD,
    V6_RETRAIN_QUICK_HOURS, V6_RETRAIN_FULL_HOURS,
    V6_MODEL_PATH, V6_REGISTRY_PATH,
    V6_TFT_CONFIG, V6_LGBM_CONFIG, V6_TCN_CONFIG, V6_REGIME_CONFIG,
    V6_TRAIN_CONFIG,
)
from data_fetcher import watch_ohlcv_forever, fetch_ohlcv_paginated, get_atr
from data.feature_engine import build_features, compute_targets
from data.normalizer import TrainValNormalizer, ExpandingNormalizer
from data.market_data import MarketDataFetcher
from models.tft_model import TFTWrapper
from models.lgbm_model import LGBMWrapper
from models.tcn_model import TCNWrapper
from models.meta_learner import MetaLearner
from models.regime_detector import RegimeDetector
from models.model_registry import ModelRegistry
from trading.decision_engine import TradeDecisionEngine
from training.train_pipeline import TrainPipeline, QuickRetrain
from training.metrics import compute_all_metrics, format_metrics_report
import trader
from trader import get_current_position, close_position, place_order_with_risk, get_all_positions
from risk_manager import check_drawdown_limit
from trade_journal import journal
import telegram_bot
from telegram_bot import (
    init_telegram, shutdown_telegram, send_message, send_photo,
    is_trading_paused, set_dependencies,
)
from training_monitor import monitor
import model as model_module
import training_monitor as monitor_module

# === Logging ===
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logger.add(
    LOG_FILE,
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level="INFO",
)

app = FastAPI(title="NN Trader v6 — Ensemble")

# === Global State ===

# Ensemble components per symbol
ensemble_models: dict = {}  # {symbol: {'tft': TFTWrapper, 'lgbm': LGBMWrapper, 'tcn': TCNWrapper}}
meta_learners: dict = {}    # {symbol: MetaLearner}
regime_detectors: dict = {} # {symbol: RegimeDetector}
normalizers: dict = {}      # {symbol: TrainValNormalizer}
decision_engine = TradeDecisionEngine()

# Model registry
registry: ModelRegistry = None

# Market data fetcher
market_data = MarketDataFetcher()

# Task tracking
ws_tasks: list = []
retrain_tasks: list = []
candle_counter: dict = {}
last_retrain_time: dict = {}  # {symbol: timestamp}
last_full_retrain_time: dict = {}


def _get_models(symbol: str) -> dict:
    """Получает или создаёт модели для символа."""
    if symbol not in ensemble_models:
        ensemble_models[symbol] = {
            'tft': TFTWrapper(V6_TFT_CONFIG),
            'lgbm': LGBMWrapper(V6_LGBM_CONFIG),
            'tcn': TCNWrapper(V6_TCN_CONFIG),
        }
        meta_learners[symbol] = MetaLearner(
            window=V6_TRAIN_CONFIG.get('meta_window', 100),
            temperature=V6_TRAIN_CONFIG.get('meta_temperature', 2.0),
        )
        regime_detectors[symbol] = RegimeDetector(V6_REGIME_CONFIG)
        normalizers[symbol] = TrainValNormalizer()
    return ensemble_models[symbol]


def _load_models(symbol: str) -> bool:
    """Загружает сохранённые модели из registry."""
    active_path = registry.get_active_path() if registry else None
    if not active_path or not Path(active_path).exists():
        return False

    models = _get_models(symbol)
    loaded = False

    for name in ['tft', 'lgbm', 'tcn']:
        model_dir = Path(active_path) / name
        if model_dir.exists():
            try:
                models[name].load(str(model_dir))
                logger.info(f"Loaded {name} for {symbol}")
                loaded = True
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")

    regime_dir = Path(active_path) / 'regime'
    if regime_dir.exists():
        try:
            regime_detectors[symbol].load(str(regime_dir))
            logger.info(f"Loaded regime detector for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to load regime detector: {e}")

    normalizer_file = Path(active_path) / 'normalizer.json'
    if normalizer_file.exists():
        try:
            normalizers[symbol].load(str(normalizer_file))
            logger.info(f"Loaded normalizer for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to load normalizer: {e}")

    return loaded


async def initial_training(symbol: str, df: pd.DataFrame):
    """
    Полное обучение ансамбля с нуля.
    Walk-Forward CV → сохранение в registry.
    """
    logger.info(f"=== INITIAL TRAINING {symbol}: {len(df)} свечей ===")
    await send_message(
        f"🧠 <b>Начинаем обучение v6</b> {symbol}\n"
        f"Данные: {len(df)} свечей\n"
        f"Модели: TFT + LightGBM + TCN\n"
        f"Валидация: Walk-Forward CV"
    )

    try:
        pipeline = TrainPipeline({
            **V6_TRAIN_CONFIG,
            'tft': V6_TFT_CONFIG,
            'lgbm': V6_LGBM_CONFIG,
            'tcn': V6_TCN_CONFIG,
            'regime': V6_REGIME_CONFIG,
            'lookback': V6_LOOKBACK,
            'target_horizon': V6_TARGET_HORIZON,
            'direction_threshold': V6_DIRECTION_THRESHOLD,
        })

        save_path = str(Path(V6_MODEL_PATH) / symbol.replace('/', '_'))
        result = pipeline.run(df, save_path=save_path)

        # Register in model registry
        summary = result['summary']
        version_id = registry.register(
            models_source_path=save_path,
            metrics=summary,
            metadata={
                'symbol': symbol,
                'data_size': len(df),
                'timestamp': datetime.utcnow().isoformat(),
            },
        )

        # Load into memory
        _load_models(symbol)

        # Store meta-learner and normalizer
        if result.get('meta_learner'):
            meta_learners[symbol] = result['meta_learner']
        if result.get('regime_detector'):
            regime_detectors[symbol] = result['regime_detector']
        if result.get('normalizer'):
            normalizers[symbol] = result['normalizer']

        await send_message(
            f"✅ <b>Обучение v6 {symbol} завершено</b>\n"
            f"Версия: {version_id}\n"
            f"TFT acc: {summary.get('tft_mean_acc', 'N/A'):.3f}\n"
            f"LGBM acc: {summary.get('lgbm_mean_acc', 'N/A'):.3f}\n"
            f"TCN acc: {summary.get('tcn_mean_acc', 'N/A'):.3f}\n"
            f"Meta-Learner: {result['meta_learner'].get_stats()}"
        )

        last_full_retrain_time[symbol] = time.time()
        logger.info(f"Training complete for {symbol}: {summary}")

    except Exception as e:
        logger.error(f"Training failed for {symbol}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        await send_message(f"❌ Ошибка обучения {symbol}: {e}")


async def quick_retrain_symbol(symbol: str, df: pd.DataFrame):
    """Быстрое дообучение на свежих данных."""
    models = _get_models(symbol)
    any_trained = any(m.is_trained for m in models.values())

    if not any_trained:
        logger.info(f"Skip quick retrain for {symbol}: no trained models")
        return

    logger.info(f"Quick retrain {symbol} on {len(df)} candles")
    try:
        qr = QuickRetrain({'lookback': V6_LOOKBACK})
        result = qr.retrain(
            df,
            tft=models['tft'],
            lgbm=models['lgbm'],
            tcn=models['tcn'],
            meta_learner=meta_learners[symbol],
            normalizer=normalizers[symbol],
        )
        last_retrain_time[symbol] = time.time()
        logger.info(f"Quick retrain {symbol} done: {result}")
    except Exception as e:
        logger.error(f"Quick retrain failed for {symbol}: {e}")


async def process_new_candle(df: pd.DataFrame, symbol: str):
    """Вызывается при получении новой свечи по WebSocket."""
    try:
        current_price = df['close'].iloc[-1]
        candle_counter[symbol] = candle_counter.get(symbol, 0) + 1
        candle_num = candle_counter[symbol]

        logger.info(
            f"Свеча #{candle_num} {symbol} | Close: {current_price:.2f} | "
            f"Буфер: {len(df)}"
        )

        models = _get_models(symbol)
        any_trained = any(m.is_trained for m in models.values())

        if not any_trained:
            logger.warning(f"{symbol}: модели не обучены, пропускаем")
            return

        # === 1. Фичи ===
        features_df = build_features(df)
        if features_df.empty or len(features_df) < V6_LOOKBACK:
            logger.warning(f"{symbol}: недостаточно данных для фичей")
            return

        # Нормализация
        normalizer = normalizers.get(symbol)
        if not normalizer or not normalizer.fitted:
            logger.warning(f"{symbol}: normalizer не готов")
            return

        X_latest = features_df.iloc[-V6_LOOKBACK:].values
        X_norm = normalizer.transform(X_latest)
        X_seq = X_norm[np.newaxis, :, :]  # (1, lookback, features)
        X_flat = X_norm[-1:, :]  # (1, features) — последний таймстеп для LGBM

        # === 2. Regime Detection ===
        detector = regime_detectors.get(symbol)
        if detector and detector.is_fitted:
            regime, regime_params = detector.predict_regime(df)
        else:
            regime = 'ranging'
            regime_params = {
                'trade': True,
                'position_scale': 0.5,
                'min_confidence': 0.65,
                'prefer_direction': 'both',
                'sl_multiplier': 2.0,
                'tp_multiplier': 3.0,
            }

        # === 3. Предсказания моделей ===
        predictions = {}

        # TFT
        tft = models['tft']
        if tft.is_trained:
            try:
                predictions['tft'] = tft.predict(X_seq[0])
            except Exception as e:
                logger.warning(f"TFT predict error: {e}")

        # LightGBM
        lgbm = models['lgbm']
        if lgbm.is_trained:
            try:
                predictions['lgbm'] = lgbm.predict(X_flat[0])
            except Exception as e:
                logger.warning(f"LGBM predict error: {e}")

        # TCN
        tcn = models['tcn']
        if tcn.is_trained:
            try:
                predictions['tcn'] = tcn.predict(X_seq[0])
            except Exception as e:
                logger.warning(f"TCN predict error: {e}")

        if not predictions:
            logger.warning(f"{symbol}: ни одна модель не дала предсказание")
            return

        # === 4. Meta-Learner ===
        meta = meta_learners[symbol]
        meta_signal = meta.combine_predictions(
            tft_pred=predictions.get('tft'),
            lgbm_pred=predictions.get('lgbm'),
            tcn_pred=predictions.get('tcn'),
        )

        # === 5. Decision Engine ===
        portfolio_state = None
        try:
            all_positions = await get_all_positions()
            portfolio_state = {
                'open_positions': len(all_positions),
                'max_positions': MAX_TOTAL_POSITIONS,
                'drawdown': 0.0,  # TODO: real drawdown from trader
            }
        except Exception:
            pass

        decision = decision_engine.should_trade(
            meta_signal=meta_signal,
            regime=regime,
            regime_params=regime_params,
            portfolio_state=portfolio_state,
        )

        action = decision['action']
        confidence = decision['confidence']
        reasons = ', '.join(decision['reasons'])

        logger.info(
            f"{symbol} | action={action} | conf={confidence:.3f} | "
            f"regime={regime} | reasons=[{reasons}]"
        )

        # === 6. Record outcome for Meta-Learner (from previous candle) ===
        # Записываем направление прошлой свечи для обновления весов
        if len(df) > 2:
            prev_ret = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            if prev_ret > V6_DIRECTION_THRESHOLD:
                actual_dir = 1
            elif prev_ret < -V6_DIRECTION_THRESHOLD:
                actual_dir = -1
            else:
                actual_dir = 0
            meta.record_all_outcomes(actual_dir, predictions)

        # === 7. Проверки и исполнение ===
        if not check_drawdown_limit(symbol):
            await send_message(f"⛔ <b>СТОП</b> {symbol}: превышен лимит просадки.")
            return

        if is_trading_paused():
            logger.info(f"{symbol} — торговля на паузе | {action}")
            return

        if action == 'hold':
            logger.info(f"{symbol} — HOLD | {reasons}")
            return

        # Проверка текущей позиции
        current_pos = await get_current_position(symbol)

        if action == 'long' and current_pos != 'long':
            if current_pos is not None:
                await close_position(symbol)

            atr = get_atr(df)
            await place_order_with_risk(symbol, 'buy', None, prediction=current_price * 1.01)
            await send_message(
                f"🚀 <b>LONG</b> {symbol} @ {current_price:.2f}\n"
                f"Conf: {confidence:.2f} | Agreement: {meta_signal['agreement']:.2f}\n"
                f"Regime: {regime} | Size: x{decision['size_multiplier']:.2f}\n"
                f"SL: x{decision.get('sl_multiplier', 2.0)} ATR | "
                f"TP: x{decision.get('tp_multiplier', 4.0)} ATR\n"
                f"Models: {list(predictions.keys())}\n"
                f"Reasons: {reasons}"
            )

        elif action == 'short' and current_pos != 'short':
            if current_pos is not None:
                await close_position(symbol)

            atr = get_atr(df)
            await place_order_with_risk(symbol, 'sell', None, prediction=current_price * 0.99)
            await send_message(
                f"🔻 <b>SHORT</b> {symbol} @ {current_price:.2f}\n"
                f"Conf: {confidence:.2f} | Agreement: {meta_signal['agreement']:.2f}\n"
                f"Regime: {regime} | Size: x{decision['size_multiplier']:.2f}\n"
                f"SL: x{decision.get('sl_multiplier', 2.0)} ATR | "
                f"TP: x{decision.get('tp_multiplier', 4.0)} ATR\n"
                f"Models: {list(predictions.keys())}\n"
                f"Reasons: {reasons}"
            )

        # === 8. Quick Retrain check ===
        now = time.time()
        last_qr = last_retrain_time.get(symbol, 0)
        if now - last_qr > V6_RETRAIN_QUICK_HOURS * 3600:
            asyncio.create_task(quick_retrain_symbol(symbol, df.tail(2000)))

        # === 9. Full Retrain check ===
        last_fr = last_full_retrain_time.get(symbol, 0)
        if now - last_fr > V6_RETRAIN_FULL_HOURS * 3600:
            asyncio.create_task(initial_training(symbol, df))

        # === 10. Периодические отчёты ===
        if candle_num % REPORT_INTERVAL_CANDLES == 0:
            stats = journal.get_statistics(symbol)
            meta_stats = meta.get_stats()
            weights = meta.get_weights()

            report = (
                f"📊 <b>Отчёт v6 {symbol}</b> (#{candle_num})\n\n"
                f"<b>Торговля:</b>\n"
                f"Сделок: {stats['total_trades']} | "
                f"WR: {stats['win_rate']:.1%} | "
                f"PnL: {stats['total_pnl']:+.2f}%\n"
                f"Max DD: {stats['max_drawdown']:.2f}%\n\n"
                f"<b>Модели:</b>\n"
                f"TFT: acc={meta_stats.get('tft', {}).get('accuracy', 0):.3f} "
                f"(w={weights.get('tft', 0):.2f})\n"
                f"LGBM: acc={meta_stats.get('lgbm', {}).get('accuracy', 0):.3f} "
                f"(w={weights.get('lgbm', 0):.2f})\n"
                f"TCN: acc={meta_stats.get('tcn', {}).get('accuracy', 0):.3f} "
                f"(w={weights.get('tcn', 0):.2f})\n\n"
                f"<b>Regime:</b> {regime}\n"
                f"<b>Registry:</b> {registry.get_stats() if registry else 'N/A'}"
            )
            await send_message(report)

    except Exception as e:
        logger.error(f"Ошибка обработки свечи {symbol}: {e}")
        import traceback
        logger.debug(traceback.format_exc())


@app.on_event("startup")
async def startup():
    global registry

    set_dependencies(
        trader_mod=trader,
        monitor_mod=monitor_module,
        model_mod=model_module,
        journal_mod=__import__('trade_journal'),
    )

    await init_telegram()

    # Initialize Model Registry
    registry = ModelRegistry(V6_REGISTRY_PATH)

    await send_message(
        "✅ <b>NN Trader v6 — Ensemble</b> запущен\n"
        "Модели: TFT + LightGBM + TCN\n"
        "Regime: HMM Detection\n"
        "Meta-Learner: Adaptive Voting\n"
        f"Макс позиций: {MAX_TOTAL_POSITIONS}\n"
        f"Quick retrain: каждые {V6_RETRAIN_QUICK_HOURS}ч\n"
        f"Full retrain: каждые {V6_RETRAIN_FULL_HOURS}ч\n"
        "Используйте /start для команд"
    )

    # Обучение для каждого символа
    for symbol in CCXT_SYMBOLS:
        try:
            logger.info(f"Загрузка данных для {symbol}...")
            df = fetch_ohlcv_paginated(symbol, total_limit=10000)
            logger.info(f"Загружено {len(df)} свечей для {symbol}")

            # Пробуем загрузить существующие модели
            _get_models(symbol)
            loaded = _load_models(symbol)

            if loaded:
                logger.info(f"Модели для {symbol} загружены из registry")
                await send_message(f"📦 Модели <b>{symbol}</b> загружены из registry v{registry.active_version}")
                # Quick retrain на свежих данных
                await quick_retrain_symbol(symbol, df.tail(2000))
            else:
                # Полное обучение с нуля
                await initial_training(symbol, df)

        except Exception as e:
            logger.error(f"Ошибка запуска {symbol}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            await send_message(f"❌ Ошибка запуска {symbol}: {e}")

    # Запускаем WebSocket
    for symbol in CCXT_SYMBOLS:
        task = asyncio.create_task(watch_ohlcv_forever(symbol, process_new_candle))
        ws_tasks.append(task)
        logger.info(f"WebSocket запущен для {symbol}")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutdown v6...")

    for task in ws_tasks:
        task.cancel()
    await asyncio.gather(*ws_tasks, return_exceptions=True)

    # Save models via registry
    for symbol in CCXT_SYMBOLS:
        models = ensemble_models.get(symbol, {})
        save_dir = Path(V6_MODEL_PATH) / symbol.replace('/', '_')
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, model in models.items():
            if model.is_trained:
                try:
                    model.save(str(save_dir / name))
                except Exception as e:
                    logger.error(f"Failed to save {name} for {symbol}: {e}")

    monitor.save_history()
    await send_message("🛑 Бот v6 остановлен")
    await shutdown_telegram()
    logger.info("Shutdown v6 завершён")


# === API Endpoints ===

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "v6-ensemble",
        "symbols": len(CCXT_SYMBOLS),
        "trading_paused": is_trading_paused(),
        "registry": registry.get_stats() if registry else None,
    }


@app.get("/stats")
async def stats():
    result = {}
    for symbol in CCXT_SYMBOLS:
        models = ensemble_models.get(symbol, {})
        meta = meta_learners.get(symbol)

        result[symbol] = {
            "journal": journal.get_statistics(symbol),
            "position": await get_current_position(symbol),
            "models_trained": {name: m.is_trained for name, m in models.items()},
            "meta_learner": meta.get_stats() if meta else None,
            "regime": regime_detectors.get(symbol, {}).predict_regime(pd.DataFrame())[0]
                      if symbol in regime_detectors else None,
        }
    return result


@app.get("/stats/{symbol}")
async def stats_symbol(symbol: str):
    ccxt_symbol = f"{symbol}/USDT"
    models = ensemble_models.get(ccxt_symbol, {})
    meta = meta_learners.get(ccxt_symbol)

    return {
        "journal": journal.get_statistics(ccxt_symbol),
        "position": await get_current_position(ccxt_symbol),
        "models_trained": {name: m.is_trained for name, m in models.items()},
        "meta_learner": meta.get_stats() if meta else None,
        "model_weights": meta.get_weights() if meta else None,
    }


@app.get("/registry")
async def registry_info():
    if not registry:
        return {"error": "Registry not initialized"}
    return {
        "stats": registry.get_stats(),
        "versions": registry.list_versions(),
    }


@app.post("/retrain/{symbol}")
async def trigger_retrain(symbol: str, full: bool = False):
    """Ручной запуск переобучения."""
    ccxt_symbol = f"{symbol}/USDT"
    try:
        df = fetch_ohlcv_paginated(ccxt_symbol, total_limit=10000)
        if full:
            asyncio.create_task(initial_training(ccxt_symbol, df))
            return {"status": "full retrain started", "symbol": ccxt_symbol}
        else:
            asyncio.create_task(quick_retrain_symbol(ccxt_symbol, df.tail(2000)))
            return {"status": "quick retrain started", "symbol": ccxt_symbol}
    except Exception as e:
        return {"error": str(e)}


@app.post("/rollback")
async def trigger_rollback(version: str = None):
    """Откат к предыдущей версии моделей."""
    if not registry:
        return {"error": "Registry not initialized"}

    success = registry.rollback(version)
    if success:
        # Reload models
        for symbol in CCXT_SYMBOLS:
            _load_models(symbol)
        return {"status": "rolled back", "active": registry.active_version}
    return {"error": "Rollback failed"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
