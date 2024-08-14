# -*- coding: utf-8 -*-

# ***************************************************
# * File        : openai_OpenAI_embedding.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-06-09
# * Version     : 0.1.060922
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

import pandas as pd
from openai import OpenAI

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


df = pd.read_csv("dataset/Reviews.csv")
df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model = 'text-embedding-3-small'))
df.to_csv('output/embedded_1k_reviews.csv', index = False)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
