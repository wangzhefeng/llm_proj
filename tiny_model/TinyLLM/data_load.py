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
import zipfile
from pathlib import Path

import pandas as pd

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def data_download(data_dir: str = "dataset/tiny_llm"):
    """
    data download
    """
    logger.info(f"Download data...")
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


def data_load(url = None):
    """
    data load

    Args:
        file_path (str): _description_
    """
    # 数据下载
    file_path = data_download()
    # logger.info(f"file_path: {file_path}")
    # 数据加载
    logger.info(f"Load data...")
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            # logger.info(f"Total number of character: {len(raw_text)}")

    return raw_text


def download_and_unzip_spam_data():
    # params
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = os.path.join(ROOT, "dataset/sms_spam_collection.zip")
    extracted_path = os.path.join(ROOT, "dataset/sms_spam_collection")
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    logger.info(f"data_file_path: {data_file_path}")
    # data file path check
    if data_file_path.exists():
        logger.info(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    # data file download
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    # unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    # add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    logger.info(f"File downloaded and saved as {data_file_path}")


def load_spam_data():
    extracted_path = os.path.join(ROOT, "dataset/sms_spam_collection")
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    # data read
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

    return df




# 测试代码 main 函数
def main():
    # ------------------------------
    # 数据加载
    # ------------------------------
    raw_text = data_load()
    logger.info(f"raw_text[:99]: {raw_text[:99]}")
    logger.info(f"raw_text[:99]: {raw_text[-99:]}")

    # ------------------------------
    # finetuning for text classification
    # ------------------------------
    download_and_unzip_spam_data()
    df = load_spam_data()
    logger.info(f"df: \n{df.head()}")

if __name__ == "__main__":
    main()
