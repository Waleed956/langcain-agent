# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

import os
from dotenv import load_dotenv

MODEL_NAME = os.getenv("minimax-m2.5:cloud")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

load_dotenv()

llm = ChatOllama(
    model="minimax-m2.5:cloud",
    temperature=0.7,

)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model= llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in lahore"}]}
)

print(response["messages"] [-1].content)