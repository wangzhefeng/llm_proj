# -*- coding: utf-8 -*-

# ***************************************************
# * File        : load_pretrained_weights.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-29
# * Version     : 1.0.012907
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

from tiny_llm.TinyLLM.utils.gtp_download import download_and_load_gpt2
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# model loading
settings, params = download_and_load_gpt2(model_size = "124M", models_dir = "gpt2")
logger.info(f"Settings: {settings}")
logger.info(f"Parameter dictionary keys: {params.keys()}")
logger.info(f"wte: {params["wte"]}")
logger.info(f"Token embedding weight tensor dimensions: {params["wte"].shape}")


# define model config in a dictionary for compactness
model_configs = {
    "gpt2-small()"
}




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
