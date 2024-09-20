# -*- coding: utf-8 -*-

# ***************************************************
# * File        : langchain_ouputparse.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-07-09
# * Version     : 0.1.070923
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

from langchain_openai import OpenAI
from langchain.schema import HumanMessage
from langchain.schema import BaseOutputParser

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


llm = OpenAI()

text = "给生产杯子的公司取三个合适的中文名字，以逗号分隔的形式输出。"
message = [HumanMessage(content = text)]


class CommaSeparatedListOutputParser(BaseOutputParser):
    """
    将 LLM 的输出内容解析为列表
    """

    def parse(self, text: str):
        """
        解析 LLM 调用的输出
        """
        return text.strip().split(",")
    



# 测试代码 main 函数
def main():
    llms_response = llm.invoke(text)
    print(CommaSeparatedListOutputParser().parse(llms_response))

if __name__ == "__main__":
    main()
