# GenAi-1-05-chat_bot

# GenAI Chat Bot

Это простой чат-бот на Python, использующий модель **Apertus-8B-Instruct-2509** от Swiss-AI через библиотеку Hugging Face Transformers.  
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

## 🚀 **Запуск**
python 1pac.py

Введите текст после Ты:
Чтобы выйти, напишите выход, exit или quit.
Ответы бота будут выводиться в консоль.
Вся переписка сохраняется в файл history.txt.

---

## 📁 Структура проекта
GenAi-1-05-chat_bot/
│
├─ chatbot.py       # Основной скрипт чат-бота
├─ history.txt      # Файл с историей диалога (создаётся после запуска)
└─ README.md        # Этот файл

---

## 💡 Примечания

Модель Apertus-8B-Instruct-2509 достаточно большая (~8B параметров), поэтому для быстрой работы рекомендуется GPU с минимум 12GB видеопамяти.
История диалога хранит последние 6 сообщений, чтобы модель учитывала контекст, но не перегружала память.


## 🔗 Ссылки
[Apertus-8B-Instruct-2509 на Hugging Face](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509?utm_source=chatgpt.com)
[Transformers Documentation](https://huggingface.co/docs/transformers)


