"""
Расширенный Telegram-бот с командами, мониторингом и графиками.
Команды:
  /start - Приветствие и список команд
  /status - Текущий статус бота и позиций
  /stats [symbol] - Статистика торговли
  /pnl - Текущий PnL по всем позициям
  /training [symbol] - Метрики обучения модели
  /chart [symbol] - График loss обучения
  /predictions [symbol] - График предсказаний
  /dashboard [symbol] - Полный дашборд
  /balance - Баланс аккаунта
  /positions - Список открытых позиций
  /model [symbol] - Информация о модели и версиях
  /rollback [symbol] - Откатить модель на предыдущую версию
  /stop - Приостановить торговлю
  /resume - Возобновить торговлю
"""

import asyncio
from datetime import datetime
from loguru import logger
from telegram import Bot, Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, CCXT_SYMBOLS

# Lazy imports — будут установлены через set_dependencies()
_trader_module = None
_monitor_module = None
_model_module = None
_journal_module = None

# Состояние бота
bot: Bot | None = None
app_instance: Application | None = None
trading_paused: bool = False
_message_queue: asyncio.Queue = asyncio.Queue()


def set_dependencies(trader_mod, monitor_mod, model_mod, journal_mod):
    """Устанавливает зависимости для избежания циклических импортов"""
    global _trader_module, _monitor_module, _model_module, _journal_module
    _trader_module = trader_mod
    _monitor_module = monitor_mod
    _model_module = model_mod
    _journal_module = journal_mod


def _get_default_symbol() -> str:
    """Возвращает первый символ из конфига"""
    return CCXT_SYMBOLS[0] if CCXT_SYMBOLS else "BTC/USDT"


def _parse_symbol(args: list) -> str:
    """Парсит символ из аргументов команды или возвращает дефолт"""
    if args:
        raw = args[0].upper()
        # Поддерживаем разные форматы: BTCUSDT, BTC/USDT, BTC
        if '/' in raw:
            return raw
        elif raw.endswith('USDT'):
            return f"{raw[:-4]}/{raw[-4:]}"
        else:
            return f"{raw}/USDT"
    return _get_default_symbol()


# === Команды бота ===

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Приветствие и список команд"""
    text = (
        "<b>🤖 NN Trader Bot v5</b>\n\n"
        "Доступные команды:\n\n"
        "<b>📊 Мониторинг:</b>\n"
        "/status — Статус бота и позиций\n"
        "/stats — Статистика торговли\n"
        "/pnl — Текущий PnL\n"
        "/balance — Баланс аккаунта\n"
        "/positions — Открытые позиции\n\n"
        "<b>🧠 Модель:</b>\n"
        "/training — Метрики обучения\n"
        "/chart — График loss\n"
        "/predictions — График предсказаний\n"
        "/dashboard — Полный дашборд\n"
        "/model — Версии модели\n"
        "/rollback — Откат модели\n\n"
        "<b>⚙️ Управление:</b>\n"
        "/stop — Приостановить торговлю\n"
        "/resume — Возобновить торговлю\n"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Текущий статус бота"""
    global trading_paused

    status_emoji = "🟢" if not trading_paused else "🔴"
    status_text = "Активен" if not trading_paused else "Приостановлен"

    lines = [f"{status_emoji} <b>Статус:</b> {status_text}\n"]

    for symbol in CCXT_SYMBOLS:
        pos = None
        if _trader_module:
            pos = await _trader_module.get_position_info(symbol)

        if pos:
            pnl_emoji = "📈" if pos['unrealized_pnl'] >= 0 else "📉"
            lines.append(
                f"\n<b>{symbol}</b>:\n"
                f"  Позиция: {pos['side'].upper()} x{pos['contracts']}\n"
                f"  Вход: {pos['entry_price']:.2f}\n"
                f"  Текущая: {pos['mark_price']:.2f}\n"
                f"  {pnl_emoji} PnL: {pos['unrealized_pnl']:+.2f} USDT\n"
                f"  Leverage: {pos['leverage']}x"
            )
        else:
            lines.append(f"\n<b>{symbol}</b>: Нет позиции")

        # Метрики обучения
        if _monitor_module:
            summary = _monitor_module.monitor.get_training_summary(symbol)
            if summary['status'] == 'active':
                trend_emoji = {
                    'decreasing': '✅', 'increasing': '⚠️', 'plateau': '➡️'
                }.get(summary.get('loss_trend', ''), '❓')
                lines.append(
                    f"  🧠 Loss: {summary['last_val_loss']:.6f} {trend_emoji}"
                )

    await update.message.reply_text('\n'.join(lines), parse_mode="HTML")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Статистика торговли"""
    symbol = _parse_symbol(context.args)

    if not _journal_module:
        await update.message.reply_text("Журнал сделок недоступен")
        return

    stats = _journal_module.journal.get_statistics(symbol)
    pred_summary = {}
    if _monitor_module:
        pred_summary = _monitor_module.monitor.get_prediction_summary(symbol)

    pnl_emoji = "📈" if stats['total_pnl'] >= 0 else "📉"

    text = (
        f"<b>📊 Статистика {symbol}</b>\n\n"
        f"Сделок: {stats['total_trades']}\n"
        f"Win Rate: {stats['win_rate']:.1%}\n"
        f"Ср. выигрыш: {stats['avg_win']:.4f}\n"
        f"Ср. проигрыш: {stats['avg_loss']:.4f}\n"
        f"{pnl_emoji} PnL: {stats['total_pnl']:+.2f}%\n"
        f"Max Drawdown: {stats['max_drawdown']:.2f}%\n"
        f"Sharpe-like: {stats['sharpe_like']:.2f}\n"
    )

    if pred_summary.get('status') == 'active':
        text += (
            f"\n<b>🎯 Точность предсказаний:</b>\n"
            f"Направление: {pred_summary['direction_accuracy']:.1%}\n"
            f"Ср. ошибка: {pred_summary['avg_error_pct']:.3f}%\n"
            f"Предсказаний: {pred_summary['count']}"
        )

    await update.message.reply_text(text, parse_mode="HTML")


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Текущий PnL по всем позициям"""
    total_pnl = 0.0
    lines = ["<b>💰 Текущий PnL</b>\n"]

    for symbol in CCXT_SYMBOLS:
        if _trader_module:
            pos = await _trader_module.get_position_info(symbol)
            if pos:
                total_pnl += pos['unrealized_pnl']
                emoji = "🟢" if pos['unrealized_pnl'] >= 0 else "🔴"
                lines.append(
                    f"{emoji} {symbol}: {pos['unrealized_pnl']:+.2f} USDT "
                    f"({pos['side'].upper()})"
                )

    if total_pnl != 0:
        total_emoji = "📈" if total_pnl >= 0 else "📉"
        lines.append(f"\n{total_emoji} <b>Итого: {total_pnl:+.2f} USDT</b>")
    else:
        lines.append("\nНет открытых позиций")

    await update.message.reply_text('\n'.join(lines), parse_mode="HTML")


async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Баланс фьючерсного и спот аккаунта"""
    if not _trader_module:
        await update.message.reply_text("Модуль торговли недоступен")
        return

    futures_bal, spot_bal = await asyncio.gather(
        _trader_module.get_balance(),
        _trader_module.get_spot_balance(),
    )
    text = (
        f"<b>💳 Баланс USDT</b>\n\n"
        f"<b>📊 Фьючерсы (demo-fapi):</b>\n"
        f"  Свободно: {futures_bal['free']:.2f}\n"
        f"  В позициях: {futures_bal['used']:.2f}\n"
        f"  Всего: {futures_bal['total']:.2f}\n\n"
        f"<b>🪙 Спот (demo-api):</b>\n"
        f"  Свободно: {spot_bal['free']:.2f}\n"
        f"  В ордерах: {spot_bal['used']:.2f}\n"
        f"  Всего: {spot_bal['total']:.2f}"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Список открытых позиций"""
    lines = ["<b>📋 Открытые позиции</b>\n"]
    has_positions = False

    for symbol in CCXT_SYMBOLS:
        if _trader_module:
            pos = await _trader_module.get_position_info(symbol)
            if pos:
                has_positions = True
                emoji = "🟢" if pos['unrealized_pnl'] >= 0 else "🔴"
                lines.append(
                    f"\n{emoji} <b>{symbol}</b>\n"
                    f"  {pos['side'].upper()} x{pos['contracts']}\n"
                    f"  Вход: {pos['entry_price']:.2f} → Сейчас: {pos['mark_price']:.2f}\n"
                    f"  PnL: {pos['unrealized_pnl']:+.2f} USDT\n"
                    f"  Leverage: {pos['leverage']}x\n"
                    f"  Ликвидация: {pos['liquidation_price']:.2f}"
                )

    if not has_positions:
        lines.append("\nНет открытых позиций")

    await update.message.reply_text('\n'.join(lines), parse_mode="HTML")


async def cmd_training(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Метрики обучения модели"""
    symbol = _parse_symbol(context.args)

    if not _monitor_module:
        await update.message.reply_text("Монитор обучения недоступен")
        return

    summary = _monitor_module.monitor.get_training_summary(symbol)

    if summary['status'] == 'no_data':
        await update.message.reply_text(f"Нет данных обучения для {symbol}")
        return

    trend_icons = {
        'decreasing': '✅ Улучшается',
        'increasing': '⚠️ Растёт',
        'plateau': '➡️ Плато',
        'unknown': '❓ Неизвестно',
    }
    overfit_icons = {
        'improving': '✅ Улучшается',
        'worsening': '⚠️ Ухудшается',
        'stable': '➡️ Стабильно',
        'unknown': '❓ Неизвестно',
    }

    text = (
        f"<b>🧠 Обучение {symbol}</b>\n\n"
        f"Шагов: {summary['total_steps']}\n"
        f"Train Loss: {summary['last_train_loss']:.6f}\n"
        f"Val Loss: {summary['last_val_loss']:.6f}\n"
        f"Min Val Loss: {summary['min_val_loss']:.6f}\n"
        f"Ср. Train: {summary['avg_train_loss']:.6f}\n"
        f"Ср. Val: {summary['avg_val_loss']:.6f}\n\n"
        f"Тренд Loss: {trend_icons.get(summary['loss_trend'], '❓')}\n"
        f"Overfitting: {overfit_icons.get(summary['overfitting_trend'], '❓')}"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def cmd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """График loss обучения"""
    symbol = _parse_symbol(context.args)

    if not _monitor_module:
        await update.message.reply_text("Монитор обучения недоступен")
        return

    chart = _monitor_module.monitor.generate_loss_chart(symbol)
    if chart is None:
        await update.message.reply_text(
            f"Недостаточно данных для графика {symbol} (нужно минимум 5 шагов обучения)"
        )
        return

    await update.message.reply_photo(
        photo=chart,
        caption=f"📈 Training Loss — {symbol}"
    )


async def cmd_predictions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """График предсказаний vs реальность"""
    symbol = _parse_symbol(context.args)

    if not _monitor_module:
        await update.message.reply_text("Монитор обучения недоступен")
        return

    chart = _monitor_module.monitor.generate_prediction_chart(symbol)
    if chart is None:
        await update.message.reply_text(
            f"Недостаточно предсказаний для графика {symbol} (нужно минимум 10)"
        )
        return

    await update.message.reply_photo(
        photo=chart,
        caption=f"🎯 Predictions — {symbol}"
    )


async def cmd_dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Полный дашборд"""
    symbol = _parse_symbol(context.args)

    if not _monitor_module:
        await update.message.reply_text("Монитор обучения недоступен")
        return

    chart = _monitor_module.monitor.generate_full_dashboard(symbol)
    if chart is None:
        await update.message.reply_text(f"Недостаточно данных для дашборда {symbol}")
        return

    await update.message.reply_photo(
        photo=chart,
        caption=f"📊 Full Dashboard — {symbol}"
    )


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Информация о модели и версиях"""
    symbol = _parse_symbol(context.args)

    if not _model_module:
        await update.message.reply_text("Модуль модели недоступен")
        return

    versions = _model_module.list_model_versions(symbol)
    meta_path = _model_module.get_model_meta_path(symbol)

    lines = [f"<b>🗂 Модель {symbol}</b>\n"]

    import os, json
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        lines.append(f"Обновлена: {meta.get('last_update', 'N/A')}")
        lines.append(f"Best Val Loss: {meta.get('best_val_loss', 'N/A')}")
        lines.append(f"Last Val Loss: {meta.get('last_val_loss', 'N/A')}")

    lines.append(f"\nВерсий сохранено: {len(versions)}")
    if versions:
        lines.append("\nПоследние 5:")
        for v in versions[-5:]:
            lines.append(f"  • {v['timestamp']} ({v['size_kb']:.0f} KB)")

    await update.message.reply_text('\n'.join(lines), parse_mode="HTML")


async def cmd_rollback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Откатить модель"""
    symbol = _parse_symbol(context.args)

    if not _model_module:
        await update.message.reply_text("Модуль модели недоступен")
        return

    success = _model_module.rollback_model(symbol, steps_back=1)
    if success:
        await update.message.reply_text(
            f"✅ Модель {symbol} откачена на предыдущую версию.\n"
            f"Модель будет загружена при следующей свече."
        )
    else:
        await update.message.reply_text(
            f"❌ Не удалось откатить модель {symbol}. Недостаточно версий."
        )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Приостановить торговлю"""
    global trading_paused
    trading_paused = True
    await update.message.reply_text(
        "🔴 <b>Торговля приостановлена</b>\n"
        "Бот продолжает обучение, но не открывает новых позиций.\n"
        "Используйте /resume для возобновления.",
        parse_mode="HTML"
    )


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Возобновить торговлю"""
    global trading_paused
    trading_paused = False
    await update.message.reply_text(
        "🟢 <b>Торговля возобновлена</b>",
        parse_mode="HTML"
    )


def is_trading_paused() -> bool:
    """Проверяет, приостановлена ли торговля"""
    return trading_paused


# === Инициализация и отправка сообщений ===

async def init_telegram():
    """Инициализация Telegram-бота с командами"""
    global bot, app_instance

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram не настроен (токен или chat_id отсутствует)")
        return

    app_instance = Application.builder().token(TELEGRAM_TOKEN).build()

    # Регистрируем команды
    app_instance.add_handler(CommandHandler("start", cmd_start))
    app_instance.add_handler(CommandHandler("status", cmd_status))
    app_instance.add_handler(CommandHandler("stats", cmd_stats))
    app_instance.add_handler(CommandHandler("pnl", cmd_pnl))
    app_instance.add_handler(CommandHandler("balance", cmd_balance))
    app_instance.add_handler(CommandHandler("positions", cmd_positions))
    app_instance.add_handler(CommandHandler("training", cmd_training))
    app_instance.add_handler(CommandHandler("chart", cmd_chart))
    app_instance.add_handler(CommandHandler("predictions", cmd_predictions))
    app_instance.add_handler(CommandHandler("dashboard", cmd_dashboard))
    app_instance.add_handler(CommandHandler("model", cmd_model))
    app_instance.add_handler(CommandHandler("rollback", cmd_rollback))
    app_instance.add_handler(CommandHandler("stop", cmd_stop))
    app_instance.add_handler(CommandHandler("resume", cmd_resume))

    # Инициализация application
    await app_instance.initialize()
    await app_instance.start()

    # Запускаем polling в фоне
    await app_instance.updater.start_polling(drop_pending_updates=True)

    bot = app_instance.bot

    # Устанавливаем меню команд
    commands = [
        BotCommand("start", "Начало и список команд"),
        BotCommand("status", "Статус бота и позиций"),
        BotCommand("stats", "Статистика торговли"),
        BotCommand("pnl", "Текущий PnL"),
        BotCommand("balance", "Баланс аккаунта"),
        BotCommand("positions", "Открытые позиции"),
        BotCommand("training", "Метрики обучения"),
        BotCommand("chart", "График loss"),
        BotCommand("predictions", "График предсказаний"),
        BotCommand("dashboard", "Полный дашборд"),
        BotCommand("model", "Версии модели"),
        BotCommand("rollback", "Откат модели"),
        BotCommand("stop", "Остановить торговлю"),
        BotCommand("resume", "Возобновить торговлю"),
    ]
    await bot.set_my_commands(commands)

    logger.info("✅ Telegram бот инициализирован с командами")


async def shutdown_telegram():
    """Graceful shutdown Telegram бота"""
    global app_instance
    if app_instance:
        try:
            await app_instance.updater.stop()
            await app_instance.stop()
            await app_instance.shutdown()
        except Exception as e:
            logger.error(f"Ошибка при остановке Telegram бота: {e}")


async def send_message(text: str):
    """Отправка сообщения в чат"""
    if bot:
        try:
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"Ошибка Telegram: {e}")
    else:
        logger.info(f"TG: {text}")


async def send_photo(photo_bytes, caption: str = ""):
    """Отправка изображения в чат"""
    if bot:
        try:
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=photo_bytes,
                caption=caption,
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"Ошибка отправки фото в Telegram: {e}")
