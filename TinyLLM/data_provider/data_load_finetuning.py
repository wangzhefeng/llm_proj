# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_load_finetuning.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-29
# * Version     : 1.0.012912
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


def download_and_unzip_spam_data():
    """
    download spam data for finetuning text classification
    """
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
    """
    load spam data for finetuning text classification
    """
    extracted_path = os.path.join(ROOT, "dataset/sms_spam_collection")
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    # data read
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

    return df




# 测试代码 main 函数
def main():
    # ------------------------------
    # finetuning for text classification
    # ------------------------------
    download_and_unzip_spam_data()
    df = load_spam_data()
    logger.info(f"df: \n{df.head()}")

if __name__ == "__main__":
    main()
