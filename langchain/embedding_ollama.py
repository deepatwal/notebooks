from langchain_ollama import OllamaEmbeddings

ollama_embeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")
 
response = ollama_embeddings.embed_query("what is the meaning of life?")

print(f"response:\n{response}")