from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser


# ollama_chat_model = ChatOllama(model="llama3.1:8b", temperature=0)
ollama_chat_model = ChatOllama(model="llama3.1:8b", temperature=0)


chat_history = [
    SystemMessage(content="You are a helpful AI assistant!")
]

str_output_parser = StrOutputParser()

while True:
    try:
        # query = input("You: ")
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
        str_output = str_output_parser.parse(response)
        print(f"str_output: {str_output}")
    except (TypeError, Exception) as e:
        print("chat model invocation:", str(e))

    if response:
        response_content = response.content
        chat_history.append(AIMessage(content=response_content))

        print(f"AI: {response_content}")


print(f"----------Message History----------")
print(chat_history)

print("Exiting...")