# -*- coding: utf-8 -*-

# ***************************************************
# * File        : api_request.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-17
# * Version     : 0.1.081716
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


def get_completion(prompt):
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "prompt": prompt,
    }
    response = requests.post(
        url = "http://127.0.0.1:6006", 
        headers = headers, 
        data = json.dumps(data)
    )

    return response.json()["response"]




# 测试代码 main 函数
def main():
    print(get_completion(prompt = "你好"))

if __name__ == "__main__":
    main()
