# -*- coding: utf-8 -*-

# ***************************************************
# * File        : openai_model_lc_api.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-20
# * Version     : 0.1.092018
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

from langchain_community.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 读取本地的环境变量
_ = load_dotenv(find_dotenv())

# 获取环境变量 OPENAI_API_KEY
openai_api_key = os.environ("OPENAI_API_KEY")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# model
# ------------------------------
# OpenAI API 密钥在环境变量中设置
llm = ChatOpenAI(temperature = 0.0)

# 手动指定 API 密钥
# llm = ChatOpenAI(temperature = 0.0, openai_api_key = "YOUR_API_KEY")

# localtest
# ----------
# output = llm.invoke("请你自我介绍以下自己！")
# print(output)


# ------------------------------
# prompt
# ------------------------------
template = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}"
human_template = "{text}"
chat_prompt = ChatPromptTemplate([
     ("system", template),
     ("human", human_template),
])

# localtest
# ----------
# text = "我带着比身体重的行李，\
# 游入尼罗河底，\
# 经过几道闪电 看到一堆光圈，\
# 不确定是不是这里。\
# "
# message = chat_prompt.format_messages(
#      input_language = "中文", 
#      output_language = "英文", 
#      text = text
# )
# print(message)
# output = llm.invoke(message)
# print(output)


# ------------------------------
# output
# ------------------------------
output_parser = StrOutputParser()

# localtest
# ----------
# output_parser.invoke(output)


# ------------------------------
# all steps
# ------------------------------
chain = chat_prompt | llm | output_parser
text = "I carried luggage heavier than my body \
    and dived into the bottom of the Nile River. \
    After passing through several flashes of lightning, \
    I saw a pile of halos, not sure if this is the place.\
"
chain.invoke({
     "input_language": "英文", 
     "output_language": "中文",
     "text": text
})



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
