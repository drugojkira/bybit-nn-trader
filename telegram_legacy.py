from telegram import Bot
from loguru import logger
import asyncio
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

bot = None

async def init_telegram():
    global bot
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        bot = Bot(token=TELEGRAM_TOKEN)
        logger.info("✅ Telegram бот инициализирован")
        await send_message("🚀 Бот запущен на Bybit Testnet")
    else:
        logger.warning("⚠️ Telegram не настроен (токен или chat_id отсутствует)")

async def send_message(text: str):
    if bot:
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Ошибка Telegram: {e}")
    else:
        logger.info(f"TG: {text}")