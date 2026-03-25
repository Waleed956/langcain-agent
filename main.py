from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

load_dotenv()

MAX_TURNS = int(os.getenv("MAX_TURNS", "5"))
MODEL_NAME = os.getenv("MODEL_NAME", "minimax-m2.5:cloud")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat history"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

chat_history = []
Max_Turns = 5


def chat(question):
    current_turns = len(chat_history) // 2

    if current_turns >= Max_Turns:
        return (
            "Context window is full. "
            "The AI may not follow your previous thread properly. "
            "Please type 'clear' for new chat"
        )

    response = chain.invoke({
        "question": question,
        "chat history": chat_history
    })

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    remaining = Max_Turns - (current_turns + 1)
    if remaining <= 2:
        response += f" (Only {remaining} turns left!)"

    return response


def main():
    print("LangChain Chatbot Ready! (type 'quit' to exit, 'clear' to reset the history)")
    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            chat_history.clear()
            print("Chat history cleared")
            continue
        print(f"AI: {chat(user_input)}")


if __name__ == "__main__":
    main()