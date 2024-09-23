# -*- coding: utf-8 -*-

# ***************************************************
# * File        : zhipuai_llm.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-04
# * Version     : 0.1.080418
# * Description : description
# * Link        : https://github.com/datawhalechina/llm-universe/blob/0ce94e828ce2fb63d47741098188544433c5e878/notebook/C4%20%E6%9E%84%E5%BB%BA%20RAG%20%E5%BA%94%E7%94%A8/zhipuai_llm.py
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from typing import Any, List, Mapping, Optional, Dict

from zhipuai import ZhipuAI
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class ZhipuAILLM(LLM):
    """
    LangChain ChatGLM:
        https://python.langchain.com/v0.2/docs/integrations/llms/chatglm/
    ChatGLM model:
        https://open.bigmodel.cn/dev/howuse/model
    """
    # 默认选用 glm-4
    model: str = "glm-4"
    # 温度系数
    temperature: float = 0.1
    # API key
    api_key: str = None

    def _call(self, 
              prompt: str, 
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        client = ZhipuAI(api_key = self.api_key)

        def gen_glm_params(prompt):
            """
            构造 GLM 模型请求参数 message

            Params:
                prompt: 对应的用户提示词
            """
            message = [{"role": "user", "content": prompt}]

            return message
        
        messages = gen_glm_params(prompt)
        response = client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = self.temperature,
        )

        if len(response.choices) > 0:
            return response.choices[0].message.content
        
        return "generate answer erro"
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """
        定义一个返回默认参数的方法
        获取调用 API 的默认参数
        """
        normal_params = {
            "temperature": self.temperature,
        }
        # print(type(self.model_kwargs))
        return {**normal_params}
    
    @property
    def _llm_type(self) -> str:
        return "Zhipu"

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
