from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# ---------------------------------------------------------------------------------
# initializing the model
# ---------------------------------------------------------------------------------

# # chat_ollama = ChatOllama(model="llama3.1:8b", temperature=0)
chat_ollama = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature=0)

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