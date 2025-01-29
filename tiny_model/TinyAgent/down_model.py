# -*- coding: utf-8 -*-

# ***************************************************
# * File        : down_model.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-25
# * Version     : 0.1.092523
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

from modelscope import snapshot_download

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# internlm model
model_dir = snapshot_download(
    "Shanghai_AI_Laboratory/internlm2-chat-20b",
    cache_dir = "./downloaded_models/",
    revision = "master",
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
