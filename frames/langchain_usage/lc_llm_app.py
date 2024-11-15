# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lc_llm_app.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-15
# * Version     : 0.1.111523
# * Description : description
# * Link        : https://python.langchain.com/docs/tutorials/llm_chain/
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
_ = load_dotenv(find_dotenv())
# notebook
# import getpass
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["OPENAI_API_KEY"] = getpass.getpass()
# script
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ------------------------------
# LM
# ------------------------------
model = ChatOpenAI(
    model = "gpt-4o-mini"
)

# ------------------------------
# prompt template
# ------------------------------
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{text}"),
    ]
)

# ------------------------------
# chaining with LCEL
# ------------------------------
chain = prompt_template | model
response = chain.invoke({
    "language": "Italian",
    "text": "hi!",
})
print(response.content)




# 测试代码 main 函数
def main():
    # llm invoke
    messages = [
        SystemMessage(content = "Translate the following from English into Italian"),
        HumanMessage(content = "hi!")
    ]
    result = model.invoke(messages)
    print(result)

    # prompt template
    result = prompt_template.invoke({
        "language": "Italian",
        "text": "hi!"
    })
    print(result)

    result_msg = result.to_messages()
    print(result_msg)

if __name__ == "__main__":
    main()
