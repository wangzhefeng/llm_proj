# -*- coding: utf-8 -*-

# ***************************************************
# * File        : openai_api.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-01
# * Version     : 0.1.080123
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
from openai import OpenAI

# 读取本地/项目的环境变量
# find_dotenv(): 寻找并定位 `.env` 文件的路基那个
# load_dotenv(): 读取 `.env` 文件，并将其中的环境变量加载到当前的运行环境中，如果设置的是环境变量，代码没有任何作用
_ = load_dotenv(find_dotenv())

# 如果需要通过代理端口访问，还需要做如下配置
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))


# ------------------------------
# method 1
# ------------------------------
completion = client.chat.completions.create(
    # 调用模型
    model = "gpt-3.5-turbo",
    # 对话列表
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello!"
        },
    ]
)
print(completion.choices[0].message.content)


# ------------------------------
# method 2
# ------------------------------
def gen_gpt_messages(prompt):
    """
    构造 GPT 模型请求参数 messages

    Params:
        prompt: 对应的用户提示词
    """
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return messages


def get_completion(prompt, model = "gpt-3.5-turbo", temperature = 0):
    """
    获取 GPT 模型调用结果

    Params:
        prompt: 对应的提示词
        model: 调用的模型，默认为 gpt-3.5-turbo，也可以按需选择 gpt-4 等其他模型
        temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~2。温度系数越低，输出内容越一致。
    """
    response = client.chat.completions.create(
        model = model,
        messages = gen_gpt_messages(prompt),
        temperature = temperature,
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    
    return "generate answer error"


get_completion("你好")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
