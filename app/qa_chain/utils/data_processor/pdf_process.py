# -*- coding: utf-8 -*-

# ***************************************************
# * File        : pdf_process.py
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
import re

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def process_pdf(pdf_page):
    """
    PDF 文档处理
    """
    pattern = re.compile(r"[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]", re.DOTALL)
    # 上下文中读取的 PDF 文件不仅将一句话按照原文的分行添加了换行符 \n，
    # 也在原本两个符号中插入了 \n，可以使用正则表达式匹配并删除掉 \n。
    pdf_page.page_content = re.sub(
        pattern,
        lambda match: match.group(0).replace("\n", ""),
        pdf_page.page_content,
    )
    # 数据中还有不少的 • 和空格，简单实用的 replace 方法即可。
    pdf_page.page_content = pdf_page.page_content.replace("•", "")
    pdf_page.page_content = pdf_page.page_content.replace(" ", "")
    
    return pdf_page




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
