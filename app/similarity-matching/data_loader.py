# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-06-09
# * Version     : 0.1.060921
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

import numpy as np
import pandas as pd
import openai
from openai import OpenAI

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# setup
pd.set_option("display.max_columns", None, "display.max_rows", None)
# open api key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


# dataset
df = pd.read_csv("dataset/Kaggle related questions on Qoura - Questions.csv")
print(df.shape)
print(df.head())


# embedding
client = OpenAI()


def get_embedding(text, model = "text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model = model).data[0].embedding


def cosine_similarity():
    pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
