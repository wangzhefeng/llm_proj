# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lc_vector_store_retrievers.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-16
# * Version     : 0.1.111602
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
_ = load_dotenv(find_dotenv())
# notebook
# import getpass
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["OPENAI_API_KEY"] = getpass.getpass()
# script
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ------------------------------
# documents
# ------------------------------
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]


# ------------------------------
# Vector store
# ------------------------------
# hf embedding model
embeddings_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

# vector store
vectorstore = Chroma.from_documents(
    documents = documents,
    embedding = embeddings_model,
)
# query
# query_res = vectorstore.similarity_search("cat")
# print(query_res)

# async query
# await vectorstore.asimilarity_search("cat")

# query return scores
# query_res = vectorstore.similarity_search_with_score("cat")
# print(query_res)

# embedding query
# embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2").embed_query("cat")
# query_res = vectorstore.similarity_search_by_vector(embedding)
# print(query_res)


# ------------------------------
# Retrievers
# ------------------------------
# method 1
# retriever = RunnableLambda(vectorstore.similarity_search).bind(k = 1)
# retr_res = retriever.batch(["cat", "shark"])
# print(retr_res)

# method 2
retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 1},
)
# retr_res = retriever.batch(["cat", "shark"])
# print(retr_res)

# example
llm = ChatHuggingFace(
    repo_id = "microsoft/Phi-3-mini-4k-instruct", 
    task = "text-generation",
    max_new_tokens = 512,
    do_sample = False,
    repetition_penalty = 1.03,
)

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages(
    [("human", message)]
)

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
response = rag_chain.invoke("tell me about cats")
print(response.content)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
