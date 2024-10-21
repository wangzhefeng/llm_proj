# -*- coding: utf-8 -*-

# ***************************************************
# * File        : doc_spliter.py
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

# 导入文本分割器
from langchain.text_splitter import RecursiveCharacterTextSplitter

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def recur_split_doc(pages, chunk_size: int = 500, overlap_size: int = 50):
    """
    * RecursiveCharacterTextSplitter 递归字符文本分割。
        将按不同的字符递归地分割(按照这个优先级 ["\n\n", "\n", " ", ""])，
        这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
    * RecursiveCharacterTextSplitter 需要关注的是四个参数：
        - separators: 分隔符字符串数组
        - chunk_size: 每个文档的字符数量限制
        - chunk_overlap: 两份文档重叠区域的长度
        - length_function: 长度计算函数

    Args:
        pages (_type_): _description_
        chunk_size (int, optional): _description_. Defaults to 500.
        overlap_size (int, optional): _description_. Defaults to 50.

    Returns:
        _type_: _description_
    """
    # 知识库中单段文本长度
    CHUNK_SIZE = chunk_size
    # 知识库中相邻文本重合长度
    OVERLAP_SIZE = overlap_size
    # 使用递归字符文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = OVERLAP_SIZE,
    )
    text_splitter.split_text(pages.page_content[0:1000])
    split_docs = text_splitter.split_documents(pages)
    print(f"切分后的文件数量：{len(split_docs)}")
    print(f"切分后的字符数（可以用来大致评估 token 数）：
          {sum([len(doc.page_content) for doc in split_docs])}")
    
    return split_docs


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
