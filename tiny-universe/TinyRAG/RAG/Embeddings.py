# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Embedding.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-06-07
# * Version     : 0.1.060721
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
from copy import copy
from typing import List, Tuple, Dict, Union, Optional

import numpy as np
from dotenv import load_dotenv, find_dotenv

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 读取本地/项目的环境变量
os.environ["CURL_CA_BUNDLE"] = ""
_ = load_dotenv(find_dotenv())
# 如果需要通过代理端口访问，还需要做如下配置
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"


class BaseEmbeddings:
    """
    Base class for Embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors

        Args:
            vector1 (List[float]): _description_
            vector2 (List[float]): _description_

        Returns:
            float: _description_
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
    
        return dot_product / magnitude


class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI Embeddings
    """
    def __init__(self, path: str = "", is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input = [text], model = model).data[0].embedding
        else:
            raise NotImplementedError


class JinaEmbedding(BaseEmbeddings):
    """
    class for Jina Embedding
    """
    def __init__(self, path: str = "jinaai/jina-embeddings-v2-base-zh", is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self._model = self.load_model()
    
    def get_embedding(self, text: str) -> List[float]:
        return self._model.encode([text])[0].tolist()
    
    def load_model(self):
        import torch
        from transformers import AutoModel
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = AutoModel.from_pretrained(self.path, trust_remote_code = True).to(device)

        return model


class ZhipuEmbedding(BaseEmbeddings):
    """
    class for Zhipu Embeddings
    """
    def __init__(self, path: str = "", is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key = os.getenv("ZHIPUAI_API_KEY"))
    
    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model = "embedding-2",
            input = text,
        )

        return response.data[0].embedding


class DashscopeEmbedding(BaseEmbeddings):
    """
    class for Dashscope Embeddings
    """
    def __init__(self, path: str = "", is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            self.client = dashscope.TextEmbedding
    
    def get_embedding(self, text: str, model: str = "text-embedding-v1") -> List[float]:
        response = self.client.call(
            model = model,
            input = text
        )

        return response.output["embeddings"][0]["embedding"]


# TODO
class BgeEmbedding(BaseEmbeddings):
    """
    class for BGE Embeddings
    """
    def __init__(self, path: str = "BAAI/bge-base-zh-v1.5", is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self._model, self._tokenizer = self.load_model(path)
    
    def get_embedding(self, text: str) -> List[float]:
        import torch
        encoded_input = self._tokenizer([text], padding = True, truncation = True, return_tensors = "pt")
        encoded_input = {k: v.to(self._model.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self._model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p = 2, dim = 1)

        return sentence_embeddings[0].tolist()
    
    def load_model(self, path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path).to(device)
        model.eval()

        return model, tokenizer




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
