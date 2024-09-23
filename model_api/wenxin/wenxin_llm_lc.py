# -*- coding: utf-8 -*-

# ***************************************************
# * File        : wenxin_llm_lc.py
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

from langchain_community.llms import QianfanLLMEndpoint

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 读取本地、项目的环境变量
_ = load_dotenv(find_dotenv())
# 获取环境变量
QIANFAN_AK = os.environ["QIANFAN_AK"]
QIANFAN_SK = os.environ["QIANFAN_SK"]
print(QIANFAN_AK)


# 模型调用
llm = QianfanLLMEndpoint(streaming = True)
response = llm("你好，请你介绍一下自己！")
print(response)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
