# Bybit NN Trader v2

**Полноценный бот с нейро-дообучением, риск-менеджментом и Telegram.**

### Добавлено:
- Fractional Kelly позиционирование
- Telegram-уведомления о сделках
- ATR-based SL/TP + Trailing Stop
- Подготовка к мульти-символам

### Запуск
```bash
cp .env.example .env
# заполни ключи и Telegram данные
pip install -r requirements.txt
python main.py
