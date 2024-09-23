# -*- coding: utf-8 -*-

# ***************************************************
# * File        : spark_model_api.py
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

from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())
# 如果需要通过代理端口访问，还需要做如下配置
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"


def gen_spark_params(model):
    """
    构造星火模型请求参数
    """
    spark_url_tpl = "wss://spark-api.xf-yun.com/{}/chat"
    model_params_dict = {
        # v1.5 版本
        "v1.5": {
            "domain": "general", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v1.1") # 云端环境的服务地址
        },
        # v2.0 版本
        "v2.0": {
            "domain": "generalv2", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v2.1") # 云端环境的服务地址
        },
        # v3.0 版本
        "v3.0": {
            "domain": "generalv3", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v3.1") # 云端环境的服务地址
        },
        # v3.5 版本
        "v3.5": {
            "domain": "generalv3.5", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v3.5") # 云端环境的服务地址
        }
    }

    return model_params_dict[model]


def gen_spark_messages(prompt):
    """
    构造星火模型请求参数 messages

    Params:
        prompt: 对应的用户提示词
    """
    messages = [
        ChatMessage(role = "user", content = prompt)
    ]

    return messages


def get_completion(prompt, model = "v3.5", temperature = 0.1):
    """
    获取星火模型调用结果

    Params:
        prompt: 对应的提示词
        model: 调用的模型，默认为 v3.5，也可以按需选择 v3.0 等其他模型
        temperature: 模型输出的温度系数，控制输出的随机程度，
            取值范围是 0~1.0，且不能设置为 0。温度系数越低，输出内容越一致。
    """
    spark_llm = ChatSparkLLM(
        spark_api_url = gen_spark_params(model)["spark_url"],
        spark_app_id = os.environ["SPARK_APPID"],
        spark_api_key = os.environ["SPARK_API_KEY"],
        spark_api_secret = os.environ["SPARK_API_SECRET"],
        spark_llm_domain = gen_spark_params(model)["domain"],
        temperature = temperature,
        streaming = False,
    )
    messages = gen_spark_messages(prompt)
    handler = ChunkPrintHandler()
    # 当 streaming设置为 False的时候, callbacks 并不起作用
    resp = spark_llm.generate([messages], callbacks=[handler])

    return resp



# 测试代码 main 函数
def main():
    get_completion("你好").generations[0][0].text

if __name__ == "__main__":
    main()
