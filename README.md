# GenAi-1-05-chat_bot

# GenAI Chat Bot

Это простой чат-бот на Python, использующий модель **microsoft/DialoGPT-medium** через библиотеку Hugging Face Transformers.  
Бот поддерживает ведение диалога и сохраняет историю переписки в файл.

---

## ⚡ Возможности

- Ввод текста пользователем через консоль
- Генерация ответов ботом с учётом последних сообщений
- Сохранение истории диалога в `history.txt`
- Работа на GPU (при наличии) или CPU

---

## 🛠 Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/MaidarTsyrenzhapov/GenAi-1-05-chat_bot.git
cd GenAi-1-05-chat_bot

⚠️ **Для работы на GPU убедитесь, что установлен torch с поддержкой CUDA**
pip install transformers torch

---

🎮 Использование
После запуска программы:
Введите ваше сообщение после приглашения Ты:
Нажмите Enter для отправки сообщения
Дождитесь ответа от бота
Для завершения работы введите: выход, exit или quit
Все сообщения автоматически сохраняются в файл history.txt в корневой директории проекта.

---

GenAi-1-05-chat_bot/
├── 1pac.py              # Основной скрипт чат-бота
├── history.txt          # Файл истории диалогов (создается автоматически)
└── README.md           # Документация проекта

---

🔧 Технические детали
Модель: microsoft/DialoGPT-medium
Архитектура: Transformer-based language model
Контекстное окно: 3 последних сообщений
Поддержка ускорения: CUDA (при наличии совместимого GPU)


## 🔗 Ссылки
[Apertus-8B-Instruct-2509 на Hugging Face](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509?utm_source=chatgpt.com)
[Transformers Documentation](https://huggingface.co/docs/transformers)


