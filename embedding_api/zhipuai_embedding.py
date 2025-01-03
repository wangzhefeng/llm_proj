# -*- coding: utf-8 -*-

# ***************************************************
# * File        : zhipuai_embedding.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-03
# * Version     : 0.1.080320
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
from dotenv import load_dotenv, find_dotenv
from typing import Dict, List, Any

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())
# 如果需要通过代理端口访问，还需要做如下配置
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"


class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """
    `Zhipuai Embeddings` embedding models.
    """
    client: Any
    """
    `zhipuai.ZhipuAI
    """

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """
        实例化 ZhipuAI 为 values["client"]

        Args:
            values (Dict): 包含配置信息的字典，必须包含 client 的字段.
        
        Returns:
            values (Dict): 包含配置信息的字典。如果环境中有 zhipuai 库，则将返回实例化的 ZhipuAI 类；
                           否则将报错 'ModuleNotFoundError: No module named 'zhipuai''.
        """
        from zhipuai import ZhipuAI
        values["client"] = ZhipuAI()
        return values
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        embeddings = self.client.embeddings.create(
            model = "embedding-2",
            input = text
        )
        return embeddings.data[0].embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        return [self.embed_query(text) for text in texts]
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronous Embed search docs.
        """
        raise NotImplementedError("Please use `embed_documents`. Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        """
        Asynchronous Embed query text.
        """
        raise NotImplementedError("Please use `aembed_query`. Official does not support asynchronous requests")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
