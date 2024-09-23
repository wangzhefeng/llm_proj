# -*- coding: utf-8 -*-

# ***************************************************
# * File        : qianfan_model_api.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-14
# * Version     : 0.1.081421
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

import qianfan

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())

# 如果需要通过代理端口访问，还需要做如下配置
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"


def gen_wenxin_messages(prompt):
    """
    构造文心模型请求参数 message

    Params:
        prompt: 对应的用户提示词
    """
    messages = [{
        "role": "user",
        "content": prompt,
    }]

    return messages


def get_completion(prompt, model = "ERNIE-Bot", temperature = 0.01):
    """
    获取文心模型调用结果

    Params:
        prompt: 对应的提示词
        model: 调用的模型，默认为 ERNIE-Bot，也可以按需选择 Yi-34B-Chat 等其他模型
        temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~1.0，
                        且不能设置为 0。温度系数越低，输出内容越一致。
    """
    chat_comp = qianfan.ChatCompletion()
    message = gen_wenxin_messages(prompt)
    resp = chat_comp.do(
        messages = message,
        model = model,
        temperature = temperature,
        system = "你是一名个人助理"
    )

    return resp["result"]


# 如果是免费用户，在使用上述函数时，可以在入参中指定一个免费的模型（例如 Yi-34B-Chat）再运行
get_completion(prompt = "你好，介绍以下你自己", model = "Yi-34B-Chat")

# 如果你有文心系列模型 ERNIE-Bot 的使用额度，则可直接运行如下函数
get_completion(prompt = "你好，介绍以下你自己")



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
