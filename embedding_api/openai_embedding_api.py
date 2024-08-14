# -*- coding: utf-8 -*-

# ***************************************************
# * File        : openai_embedding_api.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-03
# * Version     : 0.1.080314
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

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"


def openai_embedding(text: str, model: str = None):
    # 获取环境变量 OPENAI_API_KEY
    api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key = api_key)
    # embedding model
    if model == None:
        model = "text-embedding-3-small"
    # 模型调用    
    response = client.embeddings.create(
        input = text,
        model = model,
    )

    return response


response = openai_embedding(text = "要生成 embedding 的输入文本，字符串形式。")
print(f"返回的 embedding 类型为：{response.object}")
print(f"embedding 长度为：{len(response.data[0].embedding)}")
print(f"embedding (前 10) 为：{response.data[0].embedding[:10]}")
print(f"本次 embedding model 为：{response.model}")
print(f"本次 token 使用情况为：{response.usage}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
