# -*- coding: utf-8 -*-

# ***************************************************
# * File        : zhipu_embedding_api.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-03
# * Version     : 0.1.080317
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
from dotenv import load_dotenv, find_dotenv

from zhipuai import ZhipuAI

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())
# 如果需要通过代理端口访问，还需要做如下配置
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"


def zhipu_embedding(text: List[str]):
    api_key = os.environ["ZHIPUAI_API_KEY"]
    client = ZhipuAI(api_key = api_key)
    response = client.embeddings.create(
        model = "embedding-2",
        input = text,
    )
    
    return response




# 测试代码 main 函数
def main():
    text = [
        "要生成 embedding 的输入文本，字符串形式。"
    ]
    response = zhipu_embedding(text = text)

    print(f"response 类型为：{type(response)}")
    print(f"embedding 类型为：{response.object}")
    print(f"生成 embedding 的 model 为：{response.model}")
    print(f"生成的 embedding 长度为：{len(response.data[0].embedding)}")
    print(f"embedding(前 10)为: {response.data[0].embedding[:10]}")

if __name__ == "__main__":
    main()
