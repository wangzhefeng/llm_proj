# -*- coding: utf-8 -*-

# ***************************************************
# * File        : trainer.py
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
    AutoModelForSequenceClassification, 
    TrainingArguments,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer
)
from datasets import load_dataset

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", 
)

# model training args
training_args = TrainingArguments(
    output_dir = "./hf_transformers_usage/checkpoints/distilbert-base-uncased",
    learning_rate = 2e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = 2,
)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# data load
dataset = load_dataset("rotten_tomatoes")

# data preprocessing 
def tokenize_dataset(dataset):
    """
    创建一个给数据集分词的函数
    """
    return tokenizer(dataset["text"])

dataset = dataset.map(tokenize_dataset, batched = True)

# 用来从数据集中创建批次的 DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)


# trainer
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    tokenizer = tokenizer,
    data_collator = data_collator,
)

# model training
trainer.train()





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
