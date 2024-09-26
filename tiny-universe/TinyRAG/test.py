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
from LLM import OpenAIChat, InternLMChat

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
doc_dir = "./dataset"
embed_model_dir = ""
llm_dir = ""
db_storage_dir = "" 


# ------------------------------
# 建立向量数据库 
# ------------------------------
# 获得 dataset 目录下的所有文件内容并分割
docs = ReadFiles("./dataset/tiny_rag/").get_content(max_token_len = 600, cover_content = 150)
vector = VectorStore(docs)

# 创建 Embedding Model
embedding = JinaEmbedding(path = "./download_models/jinaai/jina-embedding-v2-base-zh")
vector.get_vector(EmbeddingModel = embedding)

# 将向量和文档内容保存到 storage 目录下，下次再用就可以直接加载本地的数据库
vector.persist(path = "./storage")

# ------------------------------
# 
# ------------------------------
vector = VectorStore()
vector.load_vector("./storage")

# embedding = JinaEmbedding(path = "./download_models/jinaai/jina-embedding-v2-base-zh")
question = "chrono 是什么？"
content = vector.query(question, EmbeddingModel = embedding, k = 1)[0]
print(content)

# ------------------------------
# model
# ------------------------------
model = InternLMChat(path = "./download_models/internlm/internlm2-chat-7b")
result = model.chat(question, [], content)
print(result)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
