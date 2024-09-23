# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LLM.py
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
from typing import Dict, List, Tuple, Optional, Union

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


PROMPT_TEMPLATE = {
    "RAG_PROMPT_TEMPLATE": """使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    "InternLM_PROMPT_TEMPLATE": """先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
}


class BaseModel:
    pass


class OpenAIChat(BaseModel):
    pass


class InternLMChat(BaseModel):
    pass


class DashscopeChat(BaseModel):
    pass


class ZhipuChat(BaseModel):
    pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
