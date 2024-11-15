# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-24
# * Version     : 1.0.092404
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

from RAG.utils import ReadFiles
from RAG.VectorBase import VectorStore
from RAG.Embeddings import JinaEmbedding
from RAG.LLM import InternLMChat

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# params
doc_dir = "./dataset/"
embed_model_dir = "./download_models/jinaai/jina-embedding-v2-base-zh"
llm_dir = "./downloaded_models/Shanghai_AI_Laboratory/internlm2-chat-7b"
db_storage_dir = "./storage/"


# ------------------------------
# 向量化模块
# ------------------------------
# 创建 Embedding Model
embedding = JinaEmbedding(path = embed_model_dir)


# ------------------------------
# 索引：将文档库分割成较短的 Chunk，并通过编码器构建向量索引(建立向量数据库)
# ------------------------------
# 获得 dataset 目录下的所有文件内容
docs = ReadFiles(path = doc_dir).get_content(max_token_len = 600, cover_content = 150)
vector = VectorStore(docs)
# 获得文档的词向量表示
vector.get_vector(EmbeddingModel = embedding)
# 将向量和文档内容保存到 storage 目录下，下次再用就可以直接加载本地的数据库
vector.persist(path = db_storage_dir)


# ------------------------------
# 检索：根据问题和 chunks 的相似度检索相关文档片段
# ------------------------------
# Query
question = "git 的原理是什么？"
# 根据 Query 在向量数据库中查询相关性强的文档
content = vector.query(query = question, EmbeddingModel = embedding, k = 1)[0]
# ------------------------------
# 生成：以检索到的上下文为条件，生成问题的回答
# ------------------------------
chat_model = InternLMChat(path = llm_dir)
answer = chat_model.chat(prompt = question, history = [], content = content)
print(answer)



# ------------------------------
# 从本地加载已经处理好的数据库
# ------------------------------
vector = VectorStore()
vector.load_vector(db_storage_dir)
# ------------------------------
# 检索：根据问题和 chunks 的相似度检索相关文档片段
# ------------------------------
# Query
question = "git 的原理是什么？"
# 根据 Query 在向量数据库中查询相关性强的文档
content = vector.query(query = question, EmbeddingModel = embedding, k = 1)[0]
# ------------------------------
# 生成：以检索到的上下文为条件，生成问题的回答
# ------------------------------
chat_model = InternLMChat(path = llm_dir)
answer = chat_model.chat(prompt = question, history = [], content = content)
print(answer)



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
