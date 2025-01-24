# -*- coding: utf-8 -*-

# ***************************************************
# * File        : simple_attention.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-01-25
# * Version     : 0.1.012501
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

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# tokens embeddings
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
)
logger.info(f"inputs: \n{inputs}")
logger.info(f"inputs.shape: {inputs.shape}")

# ------------------------------
# query x(2)
# ------------------------------
# attention scores(query x(2) with other x(i), i=1,...,6)
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    logger.info(f"x_i: {x_i}")
    logger.info(f"query: {query}")
    attn_scores_2[i] = torch.dot(x_i, query)
logger.info(f"attn_scores_2: \n{attn_scores_2}")
logger.info(f"attn_scores_2.shape: {attn_scores_2.shape}")

# attention weights
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
logger.info(f"attn_weights_2_tmp: \n{attn_weights_2_tmp}")
# or
def softmax_naive(x):
    return torch.exp(x) / torch.sum(torch.exp(x))
attn_weights_2_tmp = softmax_naive(attn_scores_2)
logger.info(f"attn_weights_2_tmp: \n{attn_weights_2_tmp}")
# or
attn_weights_2_tmp = torch.softmax(attn_scores_2, dim=0)
logger.info(f"attn_weights_2_tmp: \n{attn_weights_2_tmp}")

# context vector
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    logger.info(f"attn_weights_2_tmp[i]: {attn_weights_2_tmp[i]}")
    logger.info(f"x_i: {x_i}")
    context_vec_2 += attn_weights_2_tmp[i] * x_i
logger.info(f"context_vec_2: \n{context_vec_2}")

# ------------------------------
# all context vectors
# ------------------------------
# attention scores for all inputs tokens
attn_scores = torch.empty(inputs.shape[0], inputs.shape[0])
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)
# logger.info(f"attn_scores: \n{attn_scores}")
# or
attn_scores = inputs @ inputs.T
logger.info(f"attn_scores: \n{attn_scores}")

# attention weights
attn_weights = torch.softmax(attn_scores, dim = -1)
logger.info(f"attn_weights: \n{attn_weights}")

# context vectors
context_vecs = attn_weights @ inputs
logger.info(f"context_vecs: \n{context_vecs}")


# ------------------------------
# self-attention with trainable weights
# ------------------------------
# second input element
x_2 = inputs[1]
# the input embedding size, d=3
d_in = inputs.shape[1]
# the output embedding size, d=2
d_out = 2

# attention weights
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
logger.info(f"W_query: \n{W_query}")
logger.info(f"W_key: \n{W_key}")
logger.info(f"W_value: \n{W_value}")

# query, key, value vectors
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
logger.info(f"query_2: \n{query_2}")
logger.info(f"key_2: \n{key_2}")
logger.info(f"value_2: \n{value_2}")

keys = inputs @ W_key
values = inputs @ W_value
logger.info(f"keys.shape: {keys.shape}")
logger.info(f"values.shape: {values.shape}")

# unnormalized attention scores
attn_scores_2 = query_2 @ keys.T
logger.info(f"attn_scores_2: \n{attn_scores_2}")

# attention weights
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
logger.info(f"attn_weights_2: \n{attn_weights_2}")

# context vector for input query vector 2
context_vec_2 = attn_weights_2 @ values
logger.info(f"context_vec_2: \n{context_vec_2}")


# ------------------------------
# self-attention class
# ------------------------------
class SelfAttention(nn.Module):
    
    def __init__(self, d_in, d_out):
        super().__init__()

        self.W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        self.W_key = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        self.W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        # omega
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec

torch.manual_seed(123)
# the input embedding size, d=3
d_in = inputs.shape[1]
# the output embedding size, d=2
d_out = 2

sa_v1 = SelfAttention(d_in, d_out)
sa_v1_output = sa_v1(inputs)
logger.info(f"sa_v1_output: \n{sa_v1_output}")





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
