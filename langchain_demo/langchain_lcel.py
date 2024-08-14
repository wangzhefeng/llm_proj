# -*- coding: utf-8 -*-

# ***************************************************
# * File        : langchain_lcel.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-07-10
# * Version     : 0.1.071018
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

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """
    将 LLM 输出内容解析为列表
    """
    def parse(self, text: str) -> List[str]:
        """
        解析 LLM 调用的输出
        """
        return text.strip().split(",")

template = """你是一个能生成以逗号分隔的列表的助手，用户会传入一个类别，
你应该生成该类别下的 5 个对象，并以逗号分隔的形式返回。
只返回以逗号分隔的内容，不要包含其他内容。"""
human_template = "{text}"


chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])




# 测试代码 main 函数
def main():
    chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()
    print(chain.invoke({"text": "动物"}))

if __name__ == "__main__":
    main()
