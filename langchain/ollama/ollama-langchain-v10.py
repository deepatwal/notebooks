from typing import List

from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama


def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.

    Returns:
        bool: True if the user is valid, False otherwise.
    """
    # Implement validation logic here
    return True


chat_ollama = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    validate_model_on_init=True
).bind_tools([validate_user])


result = chat_ollama.invoke(
    "Could you validate user 123? They previously lived at "
    "123 Fake St in Boston MA and 234 Pretend Boulevard in "
    "Houston TX."
)

if isinstance(result, AIMessage) and result.tool_calls:
    print("Tool calls made:")
    print(result.tool_calls)
