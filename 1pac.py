from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model="swiss-ai/Apertus-8B-Instruct-2509",
    device=0
)

history = []

while True:
    user_input = input("Ты: ")
    if user_input in ["выход", "exit", "quit"]:
        break

    prompt = ""
    for i in history[-6:]:
        prompt += i + "\n"
    user = f"Ты: {user_input}\n"
    bot_prompt = "Бот:"
    prompt += user + bot_prompt

    result = chatbot(prompt)
    bot_reply = result[0]["generated_text"]

    print("Бот:", bot_reply.strip())

    history.append(f"Ты: {user_input}")
    history.append(f"Бот: {bot_reply.strip()}")

# Сохраняем историю
with open("history.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(history))   