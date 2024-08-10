from langchain_ollama import OllamaLLM

ollama_llm = OllamaLLM(model="llama3.1:8b", temperature=0)

response = ollama_llm.invoke("What is the capital of France?")

print(f"response:\n{response}")
