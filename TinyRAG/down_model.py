# -*- coding: utf-8 -*-

# ***************************************************
# * File        : down_model.py
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

from modelscope import snapshot_download

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# internlm model
model_dir = snapshot_download(
    "Shanghai_AI_Laboratory/internlm2-chat-7b",
    cache_dir = "./downloaded_models/",
    revision = "master",
)

# jinaai embedding
model_dir = snapshot_download(
    "jinaai/jina-embeddings-v2-base-zh",
    cache_dir = "./downloaded_models/",
    revision = "master"
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
