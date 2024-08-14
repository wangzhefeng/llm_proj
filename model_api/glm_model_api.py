# -*- coding: utf-8 -*-

# ***************************************************
# * File        : glm_api.py
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

from zhipuai import ZhipuAI

_ = load_dotenv(find_dotenv())

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


client = ZhipuAI(api_key = os.environ["ZHIPUAI_API_KEY"])

def get_glm_params(prompt):
    """
    构造 GLM 模型请求参数 message

    Params:
        prompt: 对应的用户提示词
    """
    message = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return message


def get_completion(prompt, model = "glm-4", temperature = 0.95):
    """
    
    """





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
