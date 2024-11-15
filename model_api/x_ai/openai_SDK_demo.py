# -*- coding: utf-8 -*-

# ***************************************************
# * File        : openai_SDK_demo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-09
# * Version     : 0.1.110900
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
from openai import OpenAI

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
_ = load_dotenv(find_dotenv())


# x.ai api key
XAI_API_KEY = os.getenv("XAI_API_KEY")

# openai client
client = OpenAI(
    api_key = XAI_API_KEY,
    base_url = "https://api.x.ai/v1",
)

# completion
completion = client.chat.completions.create(
    model = "grok-beta",
    messages=[
        {
            "role": "system", 
            "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
        },
        {
            "role": "user", 
            "content": "What is the meaning of life, the universe, and everything?"
        },
    ],
)
print(completion.choices[0].message)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
