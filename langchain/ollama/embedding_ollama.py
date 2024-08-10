# from langchain_ollama import OllamaEmbeddings

# ollama_embeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")

# response = ollama_embeddings.embed_query("what is the meaning of life?")

# print(f"response:\n{response}")


import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


web_base_loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/", ),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
        class_=("post-content", "post-title", "post-header"))),
)

docs = web_base_loader.load()

recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
documents = recursive_text_splitter.split_documents(docs)

ollama_embedding = OllamaEmbeddings(model="mxbai-embed-large:335m")

chroma_vector_store = Chroma.from_documents(
    documents=documents, embedding=ollama_embedding)



# explore the chroma vector store
# embedded_documents = chroma_vector_store.get(
#     include=["documents", "embeddings", "metadatas"])

# print(f"type(embedded_documents): {type(embedded_documents)}")
# print(f"embedded_documents.keys: {embedded_documents.keys()}")


# for id, document, embedding, metadata in zip(embedded_documents["ids"], embedded_documents["documents"], embedded_documents["embeddings"], embedded_documents["metadatas"]):
#     print(f"{id} | {document} | {embedding} | {metadata}")
