from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from typing import List, Annotated
from typing_extensions import TypedDict

import json



# ---------------------------------------------------------------------------------
# initializing the model
# ---------------------------------------------------------------------------------

# # chat_ollama = ChatOllama(model="llama3.1:8b", temperature=0)
# chat_ollama = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature=0)

# ---------------------------------------------------------------------------------
#  invocation
# ---------------------------------------------------------------------------------

# messages = [
#     SystemMessage(content="You are a helpful assistant that translates English to French. Translate the user sentence."),
#     HumanMessage(content="I love programming.")
# ]

# response = chat_ollama.invoke(messages)

# print("-"*100)
# print(f"type(response): {type(response)}")

# print("-"*100)
# print(f"response: {response}")

# print("-"*100)
# print(f"response.content: {response.content}")

# ---------------------------------------------------------------------------------
# chaining the model with prompt template
# ---------------------------------------------------------------------------------

# chat_prompt_template = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template(
#         "You are a helpful assistant that translates {input_language} to {output_language}."),
#     HumanMessagePromptTemplate.from_template("{input}")
# ])

# # formatted_prompt = chat_prompt_template.format_messages(
# #     input_language="English", output_language="French", input="hello")
# # print(f"formatted_prompt: {formatted_prompt}")

# chain = chat_prompt_template | chat_ollama

# response = chain.invoke(
#     {
#         "input_language": "English",
#         "output_language": "German",
#         "input": "I lofe programming."
#     }
# )

# print("-"*100)
# print(f"type(response): {type(response)}")

# print("-"*100)
# print(f"response: {response}")

# print("-"*50)
# print(f"response.content: {response.content}")

# ---------------------------------------------------------------------------------
# tool calling
# ---------------------------------------------------------------------------------


# class Address(TypedDict):
#     street: str
#     city: str
#     state: str


# def validate_user(user_id: int, addresses: List) -> bool:
#     """Validate user using historical addresses.

#     Args:
#         user_id: (int) the user ID.
#         addresses: Previous addresses.
#     """
#     return True


# chat_ollama = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature=0).bind_tools([validate_user])

# response = chat_ollama.invoke(
#     "Could you validate user 123? They previously lived at "
#     "123 Fake St in Boston MA and 234 Pretend Boulevard in "
#     "Houston TX."
# )
# # response.tool_calls

# print(f"-"*100)
# print(f"type(response): {type(response)}")

# print(f"-"*100)
# print(f"response:\n{response}")

# # print(f"-"*100)
# # print(f"response.response_metadata:\n{response.response_metadata=}")

# tool_call_response = response.tool_calls

# print(f"-"*100)
# print(f"type(tool_call_response): {type(tool_call_response)}")

# print(f"-"*100)
# print(f"tool_call_response:\n{tool_call_response}")

# ---------------------------------------------------------------------------------
# tool calling with decorator
# ---------------------------------------------------------------------------------

# @tool
# def validate_user(user_id: int, addresses: List) -> str:
#     """Validate user using historical addresses.

#     Args:
#         user_id: (int) the user ID.
#         addresses: Previous addresses.
#     """
#     return f"User {user_id} validated with addresses {addresses}"


# tools = [validate_user]

# chat_ollama = ChatOllama(model="llama3.1:8b-instruct-q8_0",temperature=0).bind_tools(tools)

# response = chat_ollama.invoke(
#     "Could you validate user 123? They previously lived at "
#     "123 Fake St in Boston MA and 234 Pretend Boulevard in "
#     "Houston TX."
# )

# print(f"-"*100)
# print(f"type(response): {type(response)}")

# print(f"-"*100)
# print(f"respons4e:\n{response}")

# print(f"-"*100)
# print(f"response.content:\n{response.content}")

# ---------------------------------------------------------------------------------
# tool calling in depth
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# creating a tool
# ---------------------------------------------------------------------------------
# @tool
# def multiply(a: int, b: int) -> int:
#     """Multiply two numbers."""
#     return a * b

# print(f"multiply: {multiply}")
# print(f"multiply.name: {multiply.name}")
# print(f"multiply.description: {multiply.description}")
# print(f"multiply.args: {multiply.args}")
# print(f"multiply.args_schema.schema(): {multiply.args_schema.schema()}")

# @tool
# def multiply_by_max(
#         a: Annotated[int, "scale factor"],
#         b: Annotated[List[int], "list of ints over which to take maximum"]
# ) -> int:
#     """Multiply a by the maximum in b."""
#     return a * max(b)

# print(json.dumps(multiply_by_max.args_schema.schema(), indent=4))

# ---------------------------------------------------------------------------------
# creating a tool with custom name and json args
# ---------------------------------------------------------------------------------

# class CalculatorInput(BaseModel):
#     a: int = Field(description="first number")
#     b: int = Field(description="second number")


# @tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True)
# def multiply(a: int, b: int) -> int:
#     """Multiply two numbers."""
#     return a * b

# print(f"multiply: {multiply}")
# # print(json.dumps(multiply.args_schema.schema(), indent=4))