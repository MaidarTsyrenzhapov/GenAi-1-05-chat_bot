from transformers import pipeline

def create_chatbot(model_name="microsoft/DialoGPT-medium", device=-1):
    """
    Создает чат-бота для английского диалога через pipeline.

    :param model_name: название модели для загрузки
    :param device: 0 для GPU, -1 для CPU
    :return: объект chatbot
    """
    return pipeline("text-generation", model=model_name, device=device)


def get_bot_reply(chatbot, history, user_input):
    """
    Генерирует ответ бота на основе истории сообщений.

    :param chatbot: объект chatbot
    :param history: список предыдущих сообщений
    :param user_input: новое сообщение пользователя
    :return: ответ бота
    """

    context = "\n".join(history[-3:])
    prompt = f"Bot is a friendly and helpful assistant.\n{context}\nYou: {user_input}\nBot:"
    result = chatbot(
        prompt,
        max_new_tokens=60,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=50256
    )
    bot_reply = result[0]["generated_text"].replace(prompt, "").strip()
    return bot_reply



def save_history(history, filename="history.txt"):
    """
    Сохраняет историю чата в файл.

    :param history: список сообщений
    :param filename: имя файла для сохранения
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(history))


def chat():
    """
    Запускает интерактивный английский чат с ботом.
    """
    chatbot = create_chatbot()
    history = []

    print("Type 'exit' to quit the chat.")

    while True:
        user_input = input("You: ").strip()
        if user_input in ["exit", "quit", "выход"]:
            break

        bot_reply = get_bot_reply(chatbot, history, user_input)
        print("Bot:", bot_reply)
        history.append(f"You: {user_input}")
        history.append(f"Bot: {bot_reply}")

    save_history(history)


if __name__ == "__main__":
    chat()
