# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-06-11
# * Version     : 0.1.061122
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
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# 加载分词器(若本地没有则自动下载)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-14B-Chat",
    trust_remote_code = True,
)

# 加载模型(若本地没有则自动下载)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-14B-Chat",
    # device_map 参数代表模型部署的位置，
    # auto 代表自动推断 cpu 和 gpu 个数并均匀分布，
    # 此外还可手动指定，例如"cuda:0"表示只部署在第一块显卡上
    device_map = "auto", 
    trust_remote_code = True,
).eval()

# 模型调用
response, history = model.chat(
    tokenizer, 
    "你好", 
    history = None
)
print(response)


response, history = model.chat(
    tokenizer, 
    "给我讲一个年轻人奋斗创业最终取得成功的故事。", 
    history = history
)
print(response)



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
