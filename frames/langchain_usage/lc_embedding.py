# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lc_embedding.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-16
# * Version     : 0.1.111613
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

from langchain_huggingface import HuggingFaceEmbeddings

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# hf embedding model
embeddings_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

# embed_documents
embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
print(len(embeddings), len(embeddings[0]))


# embed_query
embedded_query = embeddings_model.embed_query(
    "What was the name mentioned in the conversation?"
)
print(embedded_query)
print(len(embedded_query))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
