# -*- coding: utf-8 -*-

# ***************************************************
# * File        : sentiment_analysis2.py
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
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# device
device = "cuda" if torch.cuda.is_available() else "cpu"


model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline(
    "sentiment-analysis",
    model = model,
    tokenizer = tokenizer,
    device = device,
)
res = classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
print(res)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
