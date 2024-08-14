# -*- coding: utf-8 -*-

# ***************************************************
# * File        : wenxin_embedding_api.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-03
# * Version     : 0.1.080316
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
import json
import requests

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def wenxin_embedding(text: str):
    # 获取环境变量 wenxin_api_key, wenxin_secret_key
    api_key = os.environ["QIANFN_AK"]
    secret_key = os.environ["QIANFAN_SK"]
    # 使用 API Key、Secret Key 向 https://aip.baidubce.com/oauth/2.0/token 获取 Access token
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    payload = json.dumps("")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    response = requests.request("POST", url, headers = headers, data = payload)
    # 通过获取的 Access token 来 embedding text
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token={str(response.json().get('access_token'))}"
    input = []
    input.append(text)
    payload = json.dumps({"input": input})
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.request("POST", url, headers = headers, data = payload)

    return json.loads(response.text)


# text 应为 List(str)
text = "要生成 embedding 的输入文本，字符串形式。"
response = wenxin_embedding(text = text)

print(f"本次 embedding id 为：{response["id"]}")
print(f"本次 embedding 产生的时间戳为：{response["created"]}")
print(f"返回的 embedding 类型为：{response["object"]}")
print(f"embedding 长度为：{response["data"][0]["embedding"]}")
print(f"embedding (前 10) 为：{response["data"][0]["embedding"][:10]}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
