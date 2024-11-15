# -*- coding: utf-8 -*-

# ***************************************************
# * File        : anthropic_sdk_demo.py
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
from anthropic import Anthropic

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
_ = load_dotenv(find_dotenv())

# api key
XAI_API_KEY = os.getenv("XAI_API_KEY")

# client
client = Anthropic(
    api_key = XAI_API_KEY,
    base_url = "https://api.x.ai",
)

# completion
message = client.messages.create(
    model = "grok-beta",
    max_tokens = 128,
    system = "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy.",
    messages=[
        {
            "role": "user",
            "content": "What is the meaning of life, the universe, and everything?",
        },
    ],
)
print(message.content)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
