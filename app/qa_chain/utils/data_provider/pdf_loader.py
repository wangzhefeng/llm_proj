# -*- coding: utf-8 -*-

# ***************************************************
# * File        : pdf_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-23
# * Version     : 1.0.092319
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = [
    "load_pdf"
]

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import warnings

from langchain_community.document_loaders import PyMuPDFLoader

warnings.filterwarnings("ignore")
# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def load_pdf(doc_path: str):
    """
    加载 PDF 文档
    """
    # 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 PDF 文档路径
    loader = PyMuPDFLoader(doc_path)
    # 调用 PyMuPDFLoader Class 的函数 load 对 PDF 文件进行加载
    pdf_pages = loader.load()
    # 打印信息
    print(f"载入后的变量类型为：{type(pdf_pages)}, 该 PDF 一共包含 {len(pdf_pages)} 页。")
    
    return pdf_pages


# 测试代码 main 函数
def main():
    if sys.platform != "win32":
        doc_path = "/Users/wangzf/llm_proj/app/qa_chain/database/knowledge_lib/pumkin_book/pumpkin_book.pdf"
    else:
        doc_path = "E:/projects/llms_proj/llm_proj/app/qa_chain/database/knowledge_lib/pumkin_book/pumpkin_book.pdf" 
    pdf_pages = load_pdf(doc_path = doc_path) 
    # 第一页
    pdf_page = pdf_pages[1]
    print(
        f"每一个元素的类型：{type(pdf_page)}", 
        f"该文档的描述性数据：{pdf_page.metadata}", 
        f"查看该文档的内容：\n{pdf_page.page_content}",
        sep = "\n------\n"
    )

if __name__ == "__main__":
    main()
