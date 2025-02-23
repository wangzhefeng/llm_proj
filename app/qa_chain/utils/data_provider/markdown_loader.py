# -*- coding: utf-8 -*-

# ***************************************************
# * File        : markdown_loader.py
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
    "load_markdown"
]

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_community.document_loaders import UnstructuredMarkdownLoader

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def load_markdown(doc_path: str):
    """
    加载 Markdown 文档
    """
    # 创建一个 UnstructuredMarkdownLoader Class 实例，输入为待加载的 Markdown 文档路径
    loader = UnstructuredMarkdownLoader(doc_path)
    # 调用 UnstructuredMarkdownLoader Class 的函数 load 对 Markdown 文件进行加载
    md_pages = loader.load()
    print(f"载入后的变量类型为：{type(md_pages)}, 该 Markdown 一共包含 {len(md_pages)} 页。")
    
    return md_pages


# 测试代码 main 函数
def main():
    if sys.platform != "win32":
        doc_path = "/Users/wangzf/llm_proj/app/qa_chain/database/knowledge_lib/prompt_engineering/1. 简介 Introduction.md"
    else:
        doc_path = "E:/projects/llms_proj/llm_proj/app/qa_chain/database/knowledge_lib/prompt_engineering/1. 简介 Introduction.md"
    md_pages = load_markdown(doc_path = doc_path)
    # 第一页
    md_page = md_pages[0]
    print(
        f"每一个元素的类型：{type(md_page)}", 
        f"该文档的描述性数据：{md_page.metadata}", 
        f"查看该文档的内容：\n{md_page.page_content[0:][:200]}", 
        sep = "\n------\n"
    )

if __name__ == "__main__":
    main()
