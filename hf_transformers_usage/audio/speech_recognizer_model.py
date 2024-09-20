# -*- coding: utf-8 -*-

# ***************************************************
# * File        : speech_recognise.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-19
# * Version     : 0.1.091921
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

import torch
from transformers import pipeline
from datasets import load_dataset, Audio

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# device
device = "cuda" if torch.cuda.is_available() else "cpu"


# pipeline
speech_recognizer = pipeline(
    "automatic-speech-recognition",
    model = "facebook/wav2vec2-base-960h",
    device = device,
)

# data
dataset = load_dataset(
    "PolyAI/minds14", 
    name = "en-US", 
    split = "train",
    trust_remote_code = True,
)
# 确保加载的数据集音频采样频率与模型训练数据的采样频率一致
dataset = dataset.cast_column(
    "audio", 
    Audio(sampling_rate = speech_recognizer.feature_extractor.sampling_rate)
)
result = speech_recognizer(dataset[:4]["audio"])

print([d["text"] for d in result])




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
