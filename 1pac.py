from transformers import pipeline, AutoTokenizer
import torch


def create_chatbot(model_name="microsoft/DialoGPT-medium", device=-1):
    """
    Создает чат-бота для английского диалога через Hugging Face pipeline.

    :param model_name: str, название модели для загрузки
    :param device: int, 0 для GPU, -1 для CPU
    :return: tuple (chatbot, tokenizer) или (None, None) в случае ошибки
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        chatbot = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            device=device
        )
        return chatbot, tokenizer
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None, None


def trim_history(tokenizer, history, max_tokens=200):
    """
    Обрезает историю сообщений так, чтобы она не превышала указанное число токенов.

    :param tokenizer: объект токенизатора
    :param history: list[str], список сообщений вида ["You: ...", "Bot: ..."]
    :param max_tokens: int, максимальное количество токенов в контексте
    :return: list[str], обрезанная история
    """
    tokens = []
    trimmed = []

    for msg in reversed(history):
        msg_tokens = tokenizer.encode(msg, add_special_tokens=False)
        if len(tokens) + len(msg_tokens) > max_tokens:
            break
        tokens = msg_tokens + tokens
        trimmed.insert(0, msg)

    return trimmed


def get_bot_reply(chatbot, tokenizer, history, user_input):
    """
    Генерирует ответ бота на основе истории и нового ввода пользователя.

    :param chatbot: pipeline-объект для генерации текста
    :param tokenizer: объект токенизатора
    :param history: list[str], история сообщений
    :param user_input: str, реплика пользователя
    :return: str, ответ бота
    """
    history = trim_history(tokenizer, history, max_tokens=200)
    context = "\n".join(history)
    prompt = f"Bot is a friendly and helpful assistant.\n{context}\nYou: {user_input}\nBot:"

    try:
        result = chatbot(
            prompt,
            max_new_tokens=60,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
        )
        bot_reply = result[0]["generated_text"].replace(prompt, "").strip()
        return bot_reply
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            return "⚠️ Недостаточно памяти для генерации ответа."
        return f"⚠️ Ошибка генерации: {e}"


def save_history(history, filename="history.txt"):
    """
    Сохраняет историю чата в текстовый файл.

    :param history: list[str], история сообщений
    :param filename: str, имя файла для сохранения
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(history))


def chat():
    """
    Запускает интерактивный чат с ботом.
    Пользователь пишет реплики, бот отвечает.
    Для выхода введите 'exit', 'quit' или 'выход'.
    """
    chatbot, tokenizer = create_chatbot()
    if chatbot is None:
        print("Не удалось создать чат-бота. Выход.")
        return

    history = []
    print("Type 'exit' to quit the chat.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "выход"]:
            break

        bot_reply = get_bot_reply(chatbot, tokenizer, history, user_input)
        print("Bot:", bot_reply)
        history.append(f"You: {user_input}")
        history.append(f"Bot: {bot_reply}")

    save_history(history)


if __name__ == "__main__":
    chat()
