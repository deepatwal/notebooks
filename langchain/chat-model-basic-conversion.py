from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


ollama_chat_model = ChatOllama(model="llama3:8b", temperature=0)

# response = ollama_chat_model.invoke("Hello, ollama!")
# print(f"response: {response}")

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="what is 81 divided by 9 ?")
]

response = ollama_chat_model.invoke(messages)

print(f"\n type(response): {type(response)}")
print(f"\n response: {response}")
print(f"\n response.content: {response.content}")
