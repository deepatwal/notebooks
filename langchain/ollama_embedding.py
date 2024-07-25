# from langchain_community.embeddings import OllamaEmbeddings


# ollama_embedding = OllamaEmbeddings(model="mxbai-embed-large:335m")

# print(f"ollama_embedding.dict(): {ollama_embedding.dict()}")


# query_embeddings = ollama_embedding.embed_query("What is the capital of France?")

# print(f"type(query_embedding): {type(query_embeddings)}")
# print(f"len(query_embedding): {len(query_embeddings)}")
# print(f"query_embedding: {query_embeddings[:10]}")


# document_embeddings = ollama_embedding.embed_documents([
#     "this is a contest of the document", "this is another document"
# ])

# print(f"type(document_embedding): {type(document_embeddings)}")
# print(f"len(document_embedding): {len(document_embeddings)}")
# print(f"document_embedding: {document_embeddings[0][:10]}")


from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter

web_base_loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/", ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )
)

docs = web_base_loader.load()

# print(f"type(docs): {type(docs)}")
# print(f"docs: {docs}")

recursive_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
splits = recursive_text_splitter.split_documents(docs)

print(f"type(splits): {type(splits)}")
print(f"len(splits): {len(splits)}")
print(f"splits[:2]: {splits[:2]}")