# -*- coding: utf-8 -*-

# ***************************************************
# * File        : build_vectordb.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-22
# * Version     : 0.1.102200
# * Description : 搭建向量知识库
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import re
from dotenv import load_dotenv, find_dotenv

# Vector database
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# OpenAI Embedding
# from langchain.embeddings.openai import OpenAIEmbeddings
# 百度千帆 Embedding
# from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
# 智谱 Embedding
# from embedding_api.zhipuai_embedding import ZhipuAIEmbeddings
# M3E Embedding
from embedding_api.m3e_embedding import M3eEmbeddings

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())
# 如果需要通过代理端口访问，你需要如下配置
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'


# ------------------------------
# 文档读取、预处理、分割
# ------------------------------
# 获取 folder_path 下所有文件路径，储存在 file_paths 里
file_paths = []
folder_path = "E:/projects/llms_proj/llm_proj/app/qa_chain/database/knowledge_lib"
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths[:3])

# 遍历文件路径并把实例化的 loader 存放在 loaders 里
loaders = []
for file_path in file_paths:
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))

# 下载文件并存储到 text
texts = []
for loader in loaders: 
    texts.extend(loader.load())
# 查看数据
# text = texts[1]
# print(
#     f"每一个元素的类型：{type(text)}.", 
#     f"该文档的描述性数据：{text.metadata}", 
#     f"查看该文档的内容:\n{text.page_content[0:]}", 
#     sep="\n------\n"
# )


# ------------------------------
# 向量知识库
# ------------------------------
# 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
split_docs = text_splitter.split_documents(texts)

# 定义 Embeddings
# embedding = OpenAIEmbeddings() 
# embedding = QianfanEmbeddingsEndpoint()
# embedding = ZhipuAIEmbeddings()
embedding = M3eEmbeddings()

# 定义持久化路径
persist_directory = "E:/projects/llms_proj/llm_proj/app/qa_chain/database/vector_db/chroma"

# 构建向量知识库
vectordb = Chroma.from_documents(
    documents = split_docs,
    embedding = embedding,
    # 允许将 persist_directory 目录保存到磁盘上 
    persist_directory = persist_directory,
)
# 向量知识库持久化
vectordb.persist()
print(f"向量库中存储的数量：{vectordb._collection.count()}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
