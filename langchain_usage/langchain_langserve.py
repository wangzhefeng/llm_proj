# -*- coding: utf-8 -*-

# ***************************************************
# * File        : langchain_langserve.py
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

from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from langserve import add_routes
from fastapi import FastAPI

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# 链定义
# ------------------------------
# output parser
class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """
    将 LLM 中逗号分隔格式的输出内容解析为列表
    """

    def parse(self, text: str) -> List[str]:
        """
        解析 LLM 调用的输出
        """
        return text.strip().split(", ")
    
# prompt template
template = """你是一个能生成都好分隔的列表的助手，用户会传入一个类别，你应该生成改类别下的 5 个对象，
并以都好分隔的形式返回。
只返回一都好分隔的内容，不要包含其他内容。"""
human_template = "{text}"
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

# chain
first_chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()

# ------------------------------
# 应用定义
# ------------------------------
app = FastAPI(
    title = "第一个 LangChain 应用",
    version = "0.0.1", 
    description = "LangChain 应用接口",
)

# ------------------------------
# 添加链路由
# ------------------------------
add_routes(app, first_chain, path = "/first_app")




# 测试代码 main 函数
def main():
    import uvicorn
    uvicorn.run(app, host = "localhost", port = 8000)

    from langserve import RemoteRunnable
    remote_chain = RemoteRunnable("http://localhost:8000/first_app/")
    print(remote_chain.invoke({"text": "动物"}))

if __name__ == "__main__":
    main()
