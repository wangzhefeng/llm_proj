# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lc_agent.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-16
# * Version     : 0.1.111614
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

from dotenv import find_dotenv, load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from app.llama31_8B_Instruct.LLM import LLaMA3_1_LLM
 
# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
_ = load_dotenv(find_dotenv())
# notebook
# import getpass
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["OPENAI_API_KEY"] = getpass.getpass()
# os.environ["TAVILY_API_KEY"] = getpass.getpass()
# script
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ------------------------------
# 定义工具
# ------------------------------
# tool 1: tavily search engine
search = TavilySearchResults(max_results=2)
search_results = search.invoke("What is the weather is Shanghai")
print(search_results)

tools = [search]

# ------------------------------
# LLM
# ------------------------------
llm = LLaMA3_1_LLM(
    model_name_or_path = "E:/projects/llms_proj/llm_proj/downloaded_models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
)
print(llm("你好"))



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
