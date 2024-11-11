# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_download.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-10
# * Version     : 0.1.111012
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

from modelscope import (
    snapshot_download, 
    AutoModel, 
    AutoTokenizer
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


model_dir = snapshot_download(
    "qwen/Qwen2.5-7B-Instruct", 
    cache_dir = "downloaded_models",
    revision = "master",
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
