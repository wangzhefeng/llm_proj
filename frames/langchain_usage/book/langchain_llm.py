# -*- coding: utf-8 -*-

# ***************************************************
# * File        : llm_src.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-06-02
# * Version     : 0.1.060222
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

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

# model
# llm = OpenAI()
# chat_model = ChatOpenAI()


# template
text = "给生产杯子的公司取一个名字。"
message = [
    HumanMessage(content = text)
]




# 测试代码 main 函数
def main():
    # print(llm.invoke(text))
    # print(chat_model.invoke(message))
    pass

if __name__ == "__main__":
    main()
