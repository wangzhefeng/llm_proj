# -*- coding: utf-8 -*-

# ***************************************************
# * File        : all_minilm_L6_v2_MedQQpairs.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-21
# * Version     : 0.1.102122
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

from datasets import load_dataset
from uniem.finetuner import FineTuner

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data load and preprocess
# ------------------------------
dataset = load_dataset('vegaviazhang/Med_QQpairs')
dataset = dataset.rename_columns({
    'question1': 'sentence1', 
    'question2': 'sentence2'
})

# ------------------------------
# model finetune
# ------------------------------
finetuner = FineTuner.from_pretrained(
    'sentence-transformers/all-MiniLM-L6-v2', 
    dataset = dataset
)
fintuned_model = finetuner.run(
    epochs = 3, 
    batch_size = 32,
    output_dir = "D:/projects/llms_proj/llm_proj/embedding_api/finetuned-model/allMiniLM-medqqpairs/"
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
