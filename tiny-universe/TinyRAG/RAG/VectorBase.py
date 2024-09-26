# -*- coding: utf-8 -*-

# ***************************************************
# * File        : VectorBase.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-24
# * Version     : 1.0.092403
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
import json
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

import numpy as np
from Embeddings import (
    BaseEmbeddings, 
    OpenAIEmbedding, 
    JinaEmbedding, 
    ZhipuEmbedding
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class VectorStore:
    
    def __init__(self, document: List[str] = [""]) -> None:
        self.document = document
    
    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        """
        计算词向量

        Args:
            EmbeddingModel (BaseEmbeddings): _description_

        Returns:
            List[List[float]]: _description_
        """
        self.vectors = []
        for doc in tqdm(self.document, desc = "Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))

        return self.vectors
    
    def persist(self, path: str = "storage"):
        """
        向量数据库持久化

        Args:
            path (str, optional): _description_. Defaults to "storage".
        """
        if not os.path.exists(path):
            os.makedirs(path)
            
        with open(f"{path}/docment.json", "w", encoding = "utf-8") as f:
            json.dump(self.document, f, ensure_ascii = False)
        
        if self.vectors:
            with open(f"{path}/vectors.json", "w", encoding = "utf-8") as f:
                json.dump(self.vectors, f)
    
    def load_vector(self, path: str = "storage") -> None:
        """
        加载词向量、文档

        Args:
            path (str, optional): _description_. Defaults to "storage".
        """
        with open(f"{path}/vector.json", "r", encoding = "utf-8") as f:
            self.vectors = json.load(f)
        
        with open(f"{path}/document.json", "r", encoding = "utf-8") as f:
            self.document = json.load(f)
    
    def _get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        词向量相似性计算

        Args:
            vector1 (List[float]): _description_
            vector2 (List[float]): _description_

        Returns:
            float: _description_
        """
        return BaseEmbeddings.cosine_similarity(vector1, vector2)
    
    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        """
        检索

        Args:
            query (str): _description_
            EmbeddingModel (BaseEmbeddings): _description_
            k (int, optional): _description_. Defaults to 1.

        Returns:
            List[str]: _description_
        """
        # 将 query 计算词向量
        query_vector = EmbeddingModel.get_embedding(query)
        # 在向量数据库中检索相似性较强的词向量
        result = np.array([
            self._get_similarity(query_vector, vector) for vector in self.vectors
        ])
        # 检索到的结果
        res = np.array(self.document)[result.argsort()[-k:][::-1]].tolist()

        return res




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
