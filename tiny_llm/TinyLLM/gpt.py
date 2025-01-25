# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-25
# * Version     : 1.0.012519
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

import torch 
import torch.nn as nn

from tiny_llm.TinyLLM.activation import GELU
from utils.log_util import logger

# set options
torch.set_printoptions(sci_mode=False)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class GPTModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # Embedding
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  #TODO
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # LayerNorm
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # output head Linear
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias = False)

    def forward(self, in_idx):
        # in_idx size
        batch_size, seq_len = in_idx.shape
        # embedding
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)  # dropout
        # transformer blocks
        x = self.trf_blocks(x)
        # final norm
        x = self.final_norm(x)
        # output head
        logits = self.out_head(x)

        return logits


class TransformerBlock(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm(nn.Module):
    
    def __init__(self, emb_dim, eps = 1e-5):
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    
    def forward(self, x):
        return self.layers(x)




# 测试代码 main 函数
def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }
    
    # tokenizer
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # input data
    batch = []
    text1 = "Every effort moves you"
    text2 = "Every day hold a"
    batch.append(torch.tensor(tokenizer.encode(text1)))
    batch.append(torch.tensor(tokenizer.encode(text2)))
    batch = torch.stack(batch, dim=0)
    logger.info(f"batch: \n{batch}")
    logger.info(f"batch.shape: {batch.shape}")

    
    # ------------------------------
    # Layer Norm test
    # ------------------------------
    # data
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    logger.info(f"batch_example: \n{batch_example}")

    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    logger.info(f"out_ln: \n{out_ln}")
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    logger.info(f"Mean: \n{mean}")
    logger.info(f"Variance: \n{var}")
    
    # ------------------------------
    # Feed Forward test
    # ------------------------------
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    print(out.shape)

    # ------------------------------
    # 
    # ------------------------------
    # GPT model
    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    logger.info(f"Output: \n{logits}")
    logger.info(f"Output shape: {logits.shape}")

if __name__ == "__main__":
    main()
