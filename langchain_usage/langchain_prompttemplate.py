# -*- coding: utf-8 -*-

# ***************************************************
# * File        : langchain_prompttemplate.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-07-09
# * Version     : 0.1.070922
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

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# 基础提示模板
prompt = PromptTemplate.from_template("给生产{product}的公司取一个名字。")
res1 = prompt.format(product = "杯子")
print(res1)


# ChatPromptTemplate
template = "你是一个能够将{input_language}翻译成{output_language}的助手。"
human_template = "{text}"
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])
res2 = chat_prompt.format_messages(
    input_language = "中文",
    output_language = "英文",
    text = "我爱编程"
)
print(res2)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
