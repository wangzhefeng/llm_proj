# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-23
# * Version     : 1.0.012322
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
import warnings
from importlib.metadata import version

from utils.log_util import logger

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
logger.info(f"torch version: {version('torch')}")
logger.info(f"tiktoken version: {version('tiktoken')}")



# 测试代码 main 函数
def main():
    import tiktoken
    from tiny_llm.data_load import data_download, data_load
    from tiny_llm.tokenization import build_vocab, SimpleTokenizer

    # 训练数据下载
    file_path = data_download()
    logger.info(f"file_path: {file_path}")

    # 训练数据加载
    raw_text = data_load(file_path=file_path)
    logger.info(f"raw_text[:99]: {raw_text[:99]}")

    # 构建数据词典
    vocab = build_vocab(text=raw_text)
    logger.info(f"vocab: {vocab}")
    for i, item in enumerate(vocab.items()):
        logger.info(f"vocab[{i}]: {item}")
        if i >= 20:
            break
    
    # ------------------------------
    # tokenization
    # ------------------------------
    input_text = """It's the last he painted, you know," 
                    Mrs. Gisburn said with pardonable pride."""
    input_text2 = "Hello, do you like tea. Is this-- a test?"
    
    # method 1
    tokenizer = SimpleTokenizer(vocab=vocab)
    token_ids = tokenizer.encode(text=input_text2)
    logger.info(f"token_ids: {token_ids}")
    
    decoded_text = tokenizer.decode(token_ids=token_ids)
    logger.info(f"decoded_text: {decoded_text}")
    
    # method 2: BPE: tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    integers = tokenizer.encode(input_text, allowed_special={"<|endoftext|>"})
    logger.info(integers)
    
    strings = tokenizer.decode(integers)
    logger.info(strings)

if __name__ == "__main__":
    main()
