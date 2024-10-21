# -*- coding: utf-8 -*-

# ***************************************************
# * File        : sentence_transformer_api.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-21
# * Version     : 0.1.102123
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
from typing import List

from sentence_transformers import SentenceTransformer

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def M3EbaseEmbedding(sentences: List[str]):
    # Embedding model
    model = SentenceTransformer('moka-ai/m3e-base')
    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)
    
    return embeddings


# 测试代码 main 函数
def main():
    # Our sentences we like to encode
    sentences = [
        '* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem',
        '* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练',
        '* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one'
    ]
    # Sentences are encoded by calling model.encode()
    # embeddings = model.encode(sentences)
    embeddings = M3EbaseEmbedding(sentences = sentences)
    #Print the embeddings
    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")
        pass

if __name__ == "__main__":
    main()
