# -*- coding: utf-8 -*-

# ***************************************************
# * File        : question_answering.py
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

from transformers import pipeline

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]







# 测试代码 main 函数
def main():
    question_answerer = pipeline("question-answering")
    res = question_answerer({
        "question": "What is the name of the repository ?",
        "context": "Pipeline has been included in the huggingface/transformers repository",
    })
    print(res)

if __name__ == "__main__":
    main()
