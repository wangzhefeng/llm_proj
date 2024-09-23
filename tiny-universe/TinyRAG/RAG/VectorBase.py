# -*- coding: utf-8 -*-

# ***************************************************
# * File        : VectorBase.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-24
# * Version     : 1.0.092403
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
import json
from typing import Dict, List, Tuple, Optional, Union

from tqdm import tqdm
import numpy as np
from Embeddings import (
    BaseEmbeddings, 
    OpenAIEmbedding, 
    JinaEmbedding, 
    ZhipuEmbedding
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class VectorStore:
    pass





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
