# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-24
# * Version     : 1.0.092404
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
from dataclasses import dataclass

import torch
from tiny_transformer import Transformer

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

# 创建模型配置文件
@dataclass
class TransformerConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

model_config = TransformerConfig(
    block_size = 12, 
    vocab_size = 10, 
    n_layer = 2, 
    n_head = 4, 
    n_embd = 16, 
    dropout = 0.0, 
    bias = True,
)


# 创建模型
model = Transformer(model_config)


# 向前传递
idx = torch.randint(1, 10, (4, 8))
logits, _ = model(idx)
print(f"logits {logits.size()}")


# 模型推理
result = model.generate(idx, 3)
print(f"generate result {result.size()}")


# 生成结果
print(result)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
