# -*- coding: utf-8 -*-

# ***************************************************
# * File        : pretrained_model_usage.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-06-15
# * Version     : 0.1.061501
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

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# device
device = "cuda" if torch.cuda.is_available() else "cpu"






# 测试代码 main 函数
def main():
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
    inputs = tokenizer("Hello world!", return_tensors = "pt")
    print(inputs)
    outputs = model(**inputs)
    print(outputs)

if __name__ == "__main__":
    main()
