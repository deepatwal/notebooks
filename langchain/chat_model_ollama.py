from langchain_ollama import ChatOllama

chat_model_ollama_llm = ChatOllama(model="llama3.1:8b", temperature=0)
response = chat_model_ollama_llm.invoke("What is the capital of France?")

print(f"response:\n{response}")