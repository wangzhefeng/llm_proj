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

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())
# 如果需要通过代理端口访问，还需要做如下配置
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"


client = ZhipuAI(api_key = os.environ["ZHIPUAI_API_KEY"])


def gen_glm_params(prompt):
    '''
    构造 GLM 模型请求参数 messages

    请求参数：
        prompt: 对应的用户提示词
    '''
    messages = [{"role": "user", "content": prompt}]
    return messages


def get_completion(prompt, model="glm-4", temperature=0.95):
    '''
    获取 GLM 模型调用结果

    请求参数：
        prompt: 对应的提示词
        model: 调用的模型，默认为 glm-4，
               也可以按需选择 glm-3-turbo 等其他模型
        temperature: 模型输出的温度系数，控制输出的随机程度，
                     取值范围是 0~1.0，且不能设置为 0。
                     温度系数越低，输出内容越一致。
    '''
    messages = gen_glm_params(prompt)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
