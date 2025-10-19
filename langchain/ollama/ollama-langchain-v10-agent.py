from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_core.runnables import Runnable
from pprint import pprint


@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    punny_response: str
    weather_conditions: str | None = None


@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@tool
def get_user_location(runtime: ToolRuntime) -> str:
    """Retrieve user information based on user ID."""
    context: Context = runtime.context
    user_id = context.user_id
    return "Florida" if user_id == "1" else "SF"


checkpointer = InMemorySaver()

system_prompt = """You are an expert weather forecaster, who speaks in puns.

You have access to two only tools:

- get_user_location: Always use this to get the user's location
- get_weather_for_location: Always use this to get the weather for a specific location


ONLY use provided tools to answer.
DO NOT create your own answer
"""

chat_ollama = init_chat_model(
    model="granite4:tiny-h",
    model_provider="ollama",
    temperature=0
)


agent = create_agent(
    model=chat_ollama,
    system_prompt=system_prompt,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

pprint(response)
