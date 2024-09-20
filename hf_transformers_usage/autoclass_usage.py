# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
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

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# tokenizer
# ------------------------------
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# str
encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)

# list
pt_batch = tokenizer(
    [
        "We are very happy to show you the 🤗 Transformers library.", 
        "We hope you don't hate it."
    ],
    padding = True,
    truncation = True,
    max_length = 512,
    return_tensors = "pt",
)

# ------------------------------
# model
# ------------------------------
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)

# ------------------------------
# inference
# ------------------------------
pt_output = pt_model(**pt_batch)
print(pt_output)



# ------------------------------
# model save
# ------------------------------
pt_save_directory = "./hf_transformers_usage/pt_save_pretrained"
tokenizer.save_pretrained(pt_save_directory)
pt_model.save_pretrained(pt_save_directory)


# ------------------------------
# model load
# ------------------------------
pt_model = AutoModelForSequenceClassification.from_pretrained(
    "./hf_transformers_usage/pt_save_pretrained",
    from_tf = False,
)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
