# -*- coding: utf-8 -*-

# ***************************************************
# * File        : norm.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-12-22
# * Version     : 0.1.122214
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
import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Norm(nn.Module):
    
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        # 归一化层输入维度
        self.d_model = d_model
        # 层归一化包含两个可以学习的参数
        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))
        # 一个非常小的常数，防止分母为0
        self.eps = eps
    
    def forward(self, x):
        x_mean = x.mean(dim = -1, keepdim = True)
        x_std = x.std(dim = -1, keepdim = True) + self.eps
        norm = self.alpha * (x - x_mean) / x_std + self.bias
        
        return norm




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
