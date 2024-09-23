# -*- coding: utf-8 -*-

# ***************************************************
# * File        : spark_model_lc_api.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-23
# * Version     : 1.0.092315
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from dotenv import find_dotenv, load_dotenv

from langchain_community.llms import SparkLLM

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())
# 获取环境变量 API_KEY
IFLYTEK_SPARK_APP_ID = os.environ["IFLYTEK_SPARK_APP_ID"]
IFLYTEK_SPARK_APP_KEY = os.environ["IFLYTEK_SPARK_APP_KEY"]
IFLYTEK_SPARK_APP_SECRET = os.environ["IFLYTEK_SPARK_APP_SECRET"]


# 模型调用
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


spark_api_url = gen_spark_params(model = "v1.5")["spark_url"]
llm = SparkLLM(spark_api_url = spark_api_url) 
response = llm("你好，请你自我介绍一下自己！")
print(response)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
