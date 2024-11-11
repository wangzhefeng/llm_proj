# -*- coding: utf-8 -*-

# ***************************************************
# * File        : trans.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-10
# * Version     : 0.1.111018
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

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# 模型参数
# ------------------------------
model_dir = 'D:\projects\llms_proj\llm_proj\downloaded_models\qwen\Qwen-7B-Chat'


# ------------------------------
# 模型加载
# ------------------------------
# 加载本地 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, 
    trust_remote_code = True
)
# 加载本地半精度模型
model = AutoModelForCausalLM.from_pretrained(
    model_dir, 
    torch_dtype = torch.bfloat16,
    device_map = "auto", 
    trust_remote_code = True
).eval()
# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained(
    model_dir, 
    trust_remote_code = True
) # 可指定不同的生成长度、top_p等相关超参


# ------------------------------
# 模型推理
# ------------------------------
# 第一轮对话
response, history = model.chat(tokenizer, "你好", history = None)
print(response)

# 第二轮对话
response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history = history)
print(response)

# 第三轮对话
response, history = model.chat(tokenizer, "给这个故事起一个标题", history = history)
print(response)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
