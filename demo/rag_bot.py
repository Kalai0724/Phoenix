from rag_agent import rag_agent

print("🤖 Company Support Bot (type 'exit' to quit)\n")

while True:
    question = input("User: ")
    if question.lower() == "exit":
        break

    answer = rag_agent(question)
    print("\nBot:", answer, "\n")