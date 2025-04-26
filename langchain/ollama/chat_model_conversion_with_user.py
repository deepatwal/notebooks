from typing import List, Union
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


ollama_chat_model = ChatOllama(model="llama3.2:3b", temperature=0)

chat_history: List[Union[SystemMessage, HumanMessage, AIMessage]] = [
    SystemMessage(content="You are a helpful AI assistant!")
]

# str_output_parser = StrOutputParser()

while True:
    try:
        query = input("You: ")
    except KeyboardInterrupt as e:
        print("Exiting...")
        break

    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    response = None
    try:
        response = ollama_chat_model.invoke(input=chat_history)
    except (TypeError, Exception) as e:
        print("chat model invocation:", str(e))

    if response:
        response_content = response.content
        chat_history.append(AIMessage(content=response_content))

        print(f"AI: {response_content}")


print(f"----------Message History----------")
print(chat_history)

print("Exiting...")
