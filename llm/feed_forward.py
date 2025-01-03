# -*- coding: utf-8 -*-

# ***************************************************
# * File        : feed_forward.py
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
import torch.nn.functional as F

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff = 2048, dropout = 0.1):
        super().__init__()
       
        # d_ff 默认设置为 2048 
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
