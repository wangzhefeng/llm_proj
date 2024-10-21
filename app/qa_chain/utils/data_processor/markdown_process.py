# -*- coding: utf-8 -*-

# ***************************************************
# * File        : markdown_process.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-23
# * Version     : 1.0.092319
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

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def process_markdown(md_page):
    """
    Markdown 文档数据预处理
    """
    # 上下文中读取的 Markdown 文件每一段中间隔了一个换行符，同样可以使用 replace 方法去除。
    md_page.page_content = md_page.page_content.replace("\n\n", "\n")

    return md_page




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
