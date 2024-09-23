# -*- coding: utf-8 -*-

# ***************************************************
# * File        : wenxin_llm.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-23
# * Version     : 1.0.092315
# * Description : 基于 LangChain 定义文心模型调用方式
# * Link        : https://github.com/datawhalechina/llm-universe/blob/cace9198cf14d98c3a266da543d58bd24b07861f/notebook/C4%20%E6%9E%84%E5%BB%BA%20RAG%20%E5%BA%94%E7%94%A8/wenxin_llm.py
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
from typing import Any, List, Mapping, Optional, Dict

import qianfan
from langchain_core.callbacks.manageer import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Wenxin_LLM(LLM):
    """
    百度文心大模型接入 LangChain
    """
    # 默认选用 ERNIE-Bot-turbo 模型，即目前一般所说的百度文心大模型
    model: str = "ERNIE-Bot-turbo"
    # 温度系数
    temperature: float = 0.1
    # API_Key
    api_key: str = None
    # Secret_Key
    secret_key: str = None
    # 系统消息
    system: str = None
    
    def _call(self, 
              prompt: str, 
              stop: Optional[List[str]] = None, 
              run_manager: Optional[CallbackManagerForLLMRun]  = None,
              **kwargs: Any):
        def gen_wenxin_messages(prompt: str):
            """
            构造文心模型请求参数 message

            Args:
                prompt (str): 对应的用户提示词
            """
            messages = [{
                "role": "user",
                "content": prompt,
            }]
            return messages
        # Completion
        chat_comp = qianfan.ChatCompletion(ak = self.api_key, sk = self.secret_key)
        # message
        messages = gen_wenxin_messages(prompt)
        # result
        response = chat_comp.do(
            messages = messages,
            model = self.model,
            temperature = self.temperature,
            system = self.system,
        )
        return response["result"]
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """
        定义一个返回默认参数的方法
        获取调用 Ernie API 的默认参数
        """
        normal_params = {
            "temperature": self.temperature,
        }
        return {**normal_params}
    
    @property
    def _llm_type(self) -> str:
        return "Wenxin"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Get the identifying parameters.
        """
        return {
            **{"model": self.model}, 
            **self._default_params
        }


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
