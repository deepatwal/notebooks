from typing import List

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes

chat_ollama_model = ChatOllama(model="llama3.1:8b", temperature=0)

chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful AI assistant"),
    HumanMessagePromptTemplate.from_template("{text}")
    # HumanMessage(content="{text}")
])

str_output_parser = StrOutputParser()

ollama_chain = chat_prompt_template | chat_ollama_model | str_output_parser


app = FastAPI(
    title="LangChain server using llama3.1:8b model",
    description="A simple API server using LangChain's Runnable interfaces",
    version="1.0"
)

add_routes(app, ollama_chain, path="/chain")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)