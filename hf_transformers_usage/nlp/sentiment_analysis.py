# -*- coding: utf-8 -*-

# ***************************************************
# * File        : sentiment_analysis.py
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
from transformers import pipeline

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# device
device = "cuda" if torch.cuda.is_available() else "cpu"





# 测试代码 main 函数
def main():
    classifier = pipeline(
        "sentiment-analysis",
        device = device,
    )
    res = classifier("'We are very happy to introduce pipeline to the transformers repository.'")
    print(res)

if __name__ == "__main__":
    main()
