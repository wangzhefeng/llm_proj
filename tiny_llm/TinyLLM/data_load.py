# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-23
# * Version     : 1.0.012321
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
import urllib.request

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def data_download(data_dir: str = r"dataset\tiny_llm"):
    """
    data download

    Args:
        data_dir (str, optional): _description_. Defaults to "dataset/tiny_llm".
    """
    logger.info(f"Download data...")
    # logger.info("-" * 15)
    # 数据路径
    data_path = os.path.join(ROOT, data_dir)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    # 数据文件路径
    file_path = os.path.join(data_path, "the-verdict.txt")
    if not os.path.exists(file_path):
        url = (
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        )
        urllib.request.urlretrieve(url, file_path)
    
    logger.info(f"Data 'the-verdict.txt' has downloaded into '{data_path}'")
    
    return file_path


def data_load(file_path: str):
    """
    data load

    Args:
        file_path (str): _description_
    """
    logger.info(f"Load data...")
    # logger.info("-" * 15)
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    logger.info(f"Total number of character: {len(raw_text)}")

    return raw_text




# 测试代码 main 函数
def main():
    # 数据下载
    file_path = data_download()
    logger.info(f"file_path: {file_path}")
    
    # 数据记载
    raw_text = data_load(file_path=file_path)
    logger.info(f"raw_text[:99]: {raw_text[:99]}")

if __name__ == "__main__":
    main()
