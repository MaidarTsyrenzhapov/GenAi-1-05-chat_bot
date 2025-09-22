# Chatbot на базе Hugging Face Transformers

Этот проект — простой консольный чат-бот на английском языке, созданный с помощью библиотеки [Hugging Face Transformers](https://huggingface.co/docs/transformers/index).  
Бот использует модель [DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium), но вы можете заменить её на любую другую модель диалога.

## 🚀 Возможности
- Ведение диалога в консоли с ботом.
- Хранение истории чата (ограничение по количеству **токенов**, а не только по сообщениям).
- Безопасное получение `pad_token_id` из токенизатора (без хардкода).
- Обработка ошибок загрузки модели и ошибок генерации (в том числе `CUDA out of memory`).
- Сохранение истории чата в файл `history.txt`.

## 📦 Установка

1. Установите зависимости:
   ```bash
   pip install transformers torch


## ▶️ Запуск

Запустите скрипт:

python chatbot.py


После запуска вы увидите приглашение:

Type 'exit' to quit the chat.
You:


Введите сообщение, и бот ответит.
Для выхода используйте exit, quit или выход.

## 📝 Пример диалога
-- Type 'exit' to quit the chat.
-- You: Hello!
-- Bot: Hi there! How are you doing today?
-- You: I'm fine, thanks.
-- Bot: Glad to hear that! What would you like to talk about?



## ⚙️ Конфигурация

По умолчанию используется модель: microsoft/DialoGPT-medium.
Можно заменить её в функции create_chatbot(model_name=...).
История сообщений автоматически обрезается до 200 токенов (можно изменить в trim_history).

## 📂 Файлы

chatbot.py — основной скрипт.
history.txt — файл, куда сохраняется история после завершения диалога.
