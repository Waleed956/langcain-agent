# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "minimax-m2.5:cloud")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)


def main():
    # Run the agent
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in lahore"}]}
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()