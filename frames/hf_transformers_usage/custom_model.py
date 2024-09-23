# -*- coding: utf-8 -*-

# ***************************************************
# * File        : custom_model.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-19
# * Version     : 0.1.091922
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
from transformers import AutoConfig, AutoModel

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# device
device = "cuda" if torch.cuda.is_available() else "cpu"


# config
my_config = AutoConfig.from_pretrained(
    "distilbert/distilbert-base-uncased", 
    n_head = 12
)

# model
my_model = AutoModel.from_config(my_config)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
