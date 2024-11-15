# -*- coding: utf-8 -*-

# ***************************************************
# * File        : langchain_llm_gpu4all.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-07-16
# * Version     : 0.1.071622
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
import io
import requests
from tqdm import tqdm
from pydantic import Field
from typing import List, Mapping, Optional, Any

from langchain.llms.base import LLM
from gpt4all import GPT4All

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class CustomLLM(LLM):
    """
    一个自定义的 LLM 类，用于集成 GPT4All 模型

    参数：
        model_folder_path: (str) 存放模型的文件夹路径
        model_name: (str) 要使用的模型名称(<模型名称>.bin)
        allow_download: (bool) 是否允许下载模型

        backend: (str) 模型的后端(支持的后端: llama/gptj)
        n_batch: (int) 
        n_threads: (int) 要使用的线程数
        n_predict: (int) 要生成的最大 token 数
        temp: (float) 用于采样的温度
        top_p: (float) 用于采样的 top_p 值
        top_k: (int) 用于采样的 top_k 值
    """
    # 以下是类属性的定义
    model_folder_path: str = Field(None, alias = "model_folder_path")
    model_name: str = Field(None, alias = "model_name")
    allow_download: bool = Field(None, alias = "allow_download")

    # 所有可选参数
    backend: Optional[str] = "llama" 
    n_batch: Optional[int] = 8
    n_threads: Optional[int] = 4
    n_predict: Optional[int] = 256
    temp: Optional[float] = 0.7
    top_p: Optional[float] = 0.1
    top_k: Optional[int] = 40

    # 初始化模型实例
    gpt4_model_instance: Any = None

    def __init__(self, model_folder_path, model_name, allow_download, **kwargs):
        super(CustomLLM, self).__init__()
        # 类构造函数的实现
        self.model_folder_path: str = model_folder_path
        self.model_name: str = model_name
        self.allow_download: bool = allow_download
        # 触发自动下载
        self.auto_download()
        # 创建 GPT4All 模型实例
        self.gpt4_model_instance = GPT4All(
            model_name = self.model_name,
            model_path = self.model_folder_path,
        )
    
    def auto_download(self) -> None:
        """
        此方法将会下载模型到指定路径
        """
        ...

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        返回一个字典类型，包含 LLM 的唯一标识
        """
        return {
            "model_name": self.model_name,
            "model_path": self.model_folder_path,
            **self._get_model_default_parameters
        }
    
    @property
    def _llm_type(self) -> str:
        """
        它告诉我们正在使用什么类型的 LLM
        例如：这里将使用 GPT4All 模型
        """
        return "gpt4all"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        这是主要的方法，将在我们使用 LLM 时调用
        重写基类方法，根据用户输入的 prompt 来响应用户，返回字符串。

        Args:
            prompt (str): _description_
            stop (Optional[List[str]], optional): _description_. Defaults to None.
        """
        params = {
            **self._get_model_default_parameters,
            **kwargs,
        }
        # 使用 GPT-4 模型实例开始一个聊天会话
        with self.gpt4_model_instance.chat_session():
            # 生成响应：根据输入的提示词(prompt)和参数(params)生成响应
            response_generator = self.gpt4_model_instance.generate(prompt, **params)
            # 判断是否是流式响应模式
            if params["streaming"]:
                # 创建一个字符串 IO 流来暂存响应数据
                response = io.StringIO()
                for token in response_generator:
                    # 遍历生成器生成的每个令牌(token)
                    print(token, end = "", flush = True)
                    response.write(token)
                response_message = response.getvalue()
                response.close()
                return response_message
            # 如果不是流式响应模式，直接返回响应生成器
            return response_generator



# 测试代码 main 函数
def main():
    def test(a, b):
        a = 1
        b = 2
        c = a + b
        return c
        ...

    res = test(1, 2)
    print(res)

if __name__ == "__main__":
    main()
