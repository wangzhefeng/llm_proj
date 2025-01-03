# -*- coding: utf-8 -*-

# ***************************************************
# * File        : multi_head_attention.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-12-22
# * Version     : 0.1.122213
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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class MultiHeadAttention(nn.Module):
    
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model  # TODO
        self.d_k = d_model // heads  # TODO
        self.h = heads  # TODO

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(q, k, v, d_k, mask = None, dropout = None):
        # scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # 掩盖掉那些为了填补长度增加的单元，使其通过 softmax 计算后为 0
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        # softmax
        scores = F.softmax(scores, dim = -1)
        # dropout
        if dropout is not None:
            scores = dropout(scores) 
        # output
        output = torch.matmul(scores, v)

        return output

    def forward(self, q, k, v, mask = None):
        bs = q.size(0)
        # 进行线性操作划分成 h 个 head
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # 矩阵转置
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # 计算 attention
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        # 连接多个 head
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        # output
        output = self.out(concat)

        return output



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
