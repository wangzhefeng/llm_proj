# -*- coding: utf-8 -*-

# ***************************************************
# * File        : chorma_utils.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-23
# * Version     : 1.0.092320
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 使用 OpenAI Embedding
# from langchain.embeddings.openai import OpenAIEmbeddings
# 使用百度千帆 Embedding
# from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
# 使用自己封装的智谱 Embedding，需要将封装代码下载到本地使用
from embedding_api.zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def chorma_save(split_docs, embedding, persist_directory: str = '../../data_base/vector_db/chroma'):
    """
    TODO

    Args:
        split_docs (_type_): _description_
        embedding (_type_): _description_
        persist_directory (str, optional): 定义持久化路径. Defaults to '../../data_base/vector_db/chroma'.
    """
    vectordb = Chroma.from_documents(
        documents = split_docs,
        embedding = embedding,
        persist_directory = persist_directory  # 允许将 persist_directory 目录保存到磁盘上
    )
    vectordb.persist()
    print(f"向量库中存储的数量：{vectordb._collection.count()}")
    
    return vectordb


def chorma_query(vectordb, question: str = "什么是大语言模型"):
    """
    TODO

    Args:
        vectordb (_type_): _description_
        question (str, optional): _description_. Defaults to "什么是大语言模型".
    """
    # 余弦相似度搜索
    sim_docs = vectordb.similarity_search(question, k = 3)
    print(f"检索到的内容数：{len(sim_docs)}")
    for i, sim_doc in enumerate(sim_docs):
        print(f"SIM 检索到的第{i}个内容：\n{sim_doc.page_content[:200]}", end = "\n-------------\n")
    
    # 最大边际相关性(MMR, Maximum Marginal Relevance)搜索
    mmr_docs = vectordb.max_marginal_relevance_search(question, k = 3)
    print(f"检索到的内容数：{len(sim_docs)}")
    for i, sim_doc in enumerate(mmr_docs):
        print(f"MMR 检索到的第 {i} 个内容：\n{sim_doc.page_content[:200]}", end = "\n-----------\n")

    return sim_docs, mmr_docs




# 测试代码 main 函数
def main():
    # 定义 Embeddings
    # embedding = OpenAIEmbeddings() 
    # embedding = QianfanEmbeddingsEndpoint()
    embedding = ZhipuAIEmbeddings()

if __name__ == "__main__":
    main()
