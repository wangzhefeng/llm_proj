# -*- coding: utf-8 -*-

# ***************************************************
# * File        : embedding.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-24
# * Version     : 1.0.012400
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]







# 测试代码 main 函数
def main():
    import torch

    # input example(after tokenization)
    input_ids = torch.tensor([2, 3, 5, 1])
    # vocabulary of 6 words
    vocab_size = 6
    # embedding size 3
    output_dim = 3
    # embedding layer
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=output_dim)
    logger.info(embedding_layer.weight)
    logger.info(embedding_layer(torch.tensor([3])))
    logger.info(embedding_layer(input_ids))

if __name__ == "__main__":
    main()
