# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils.py
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
import re
import json
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union

import PyPDF2
import markdown
import html2text
import tiktoken
from bs4 import BeautifulSoup

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class ReadFiles:
    """
    class to read files.
    """

    def __init__(self, path: str) -> None:
        pass
    
    def get_files(self):
        pass
    
    def get_content(self):
        pass
    
    @classmethod
    def get_chunk(cls):
        pass
    
    @classmethod
    def read_file_content(cls):
        pass
    
    @classmethod
    def read_pdf(cls):
        pass
    
    @classmethod
    def read_markdown(cls):
        pass
    
    @classmethod
    def read_text(cls):
        pass
    

class Documents:
    """
    获取已经分好类的 JSON 格式文档
    """
    
    def __init__(self, path: str = "") -> None:
        self.path = path

    def get_content(self):
        with open(self.path, mode = "r", encoding = "utf-8") as f:
            content = json.load(f)
        
        return content




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
