# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-24
# * Version     : 1.0.092403
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = [
    "ReadFiles",
    "Documents",
]

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import re
import json
from typing import List

import PyPDF2
import markdown
import html2text
import tiktoken
from bs4 import BeautifulSoup

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    """
    class to read files.
    """

    def __init__(self, path: str) -> None:
        """
        Args:
            path (str): 目标文件夹路径
        """
        self._path = path
        self.file_list = self.get_files()
    
    def get_files(self) -> List:
        """
        获取指定文件夹下的文件

        Returns:
            List: 指定文件夹下的文件名
        """
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".md"):
                    # 如何满足要求，将其绝对路径加入到结果列表
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        
        return file_list
    
    def get_content(self, max_token_len: int = 600, cover_content: int =  150) -> List:
        """
        读取文件内容

        Args:
            max_token_len (int, optional): TODO. Defaults to 600.
            cover_content (int, optional): TODO. Defaults to 150.

        Returns:
            List: 文档内容列表
        """
        docs = []
        for file in self.file_list:
            content = self._read_file_content(file)
            chunk_content = self._get_chunk(
                content, 
                max_token = max_token_len, 
                cover_content = cover_content
            )
            docs.extend(chunk_content)

        return docs
    
    @classmethod
    def _read_file_content(cls, file_path: str) -> str:
        """
        根据文件扩展名选择读取方法

        Args:
            file_path (str): _description_

        Returns:
            str: _description_
        """
        if file_path.endswith(".pdf"):
            return cls._read_pdf(file_path)
        elif file_path.endswith(".md"):
            return cls._read_markdown(file_path)
        elif file_path.endswith(".txt"):
            return cls._read_text(file_path)
        else:
            raise ValueError("Unsupported file type")
    
    @classmethod
    def _get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150) -> List:
        """
        分割文档为 chunk

        Args:
            text (str): 待分割的文档
            max_token_len (int, optional): 最大 token 长度. Defaults to 600.
            cover_content (int, optional): token 重叠长度. Defaults to 150.

        Returns:
            (List): 分割好的 chunk
        """
        # chunk text
        chunk_text = []
        # TODO
        curr_len = 0
        curr_chunk = ""
        # token len
        token_len = max_token_len - cover_content
        
        # text split
        lines = text.splitlines()  # 假设以换行符分割文本为行
        # chunk split
        for line in lines:
            # 处理 line 中的空格
            line = line.replace(" ", "")
            # TODO
            line_len = len(enc.encode(line))
            # 如果单行长度就超过限制，则将其分割成多个块
            if line_len > max_token_len:
                num_chunks = (line_len + token_len - 1) // token_len
                for i in range(num_chunks):
                    start = i * token_len
                    end = start + token_len
                    # 避免跨单词分割
                    while not line[start:end].rstrip().isspace():
                        start += 1
                        end += 1
                        if start >= line_len:
                            break
                    curr_chunk = curr_chunk[-cover_content:] + line[start:end]
                    chunk_text.append(curr_chunk)
                
                # 处理最后一个块
                start = (num_chunks - 1) * token_len
                curr_chunk = curr_chunk[-cover_content:] + line[start:end]
                chunk_text.append(curr_chunk) 
            
            if curr_len + line_len <= token_len:
                curr_chunk += line
                curr_chunk += "\n"
                curr_len += line_len
                curr_len += 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = curr_chunk[-cover_content:] + line
                curr_len = line_len + cover_content

        if curr_chunk:
            chunk_text.append(curr_chunk)
        
        return chunk_text
    
    @classmethod
    def _read_pdf(cls, file_path: str) -> str:
        """
        读取 PDF 文件

        Args:
            file_path (str): 文件路径

        Returns:
            str: PDF 文件内容
        """
        with open(file_path, "r") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()

            return text
    
    @classmethod
    def _read_markdown(cls, file_path: str) -> str:
        """
        读取 Markdown 文件

        Args:
            file_path (str): 文件路径

        Returns:
            str: Markdown 文件内容
        """
        with open(file_path, "r", encoding = "utf-8") as file:
            md_text = file.read()
            html_text = markdown.markdown(md_text)
            # 使用 BeautifulSoup 从 HTML 中提取纯文本
            soup = BeautifulSoup(html_text, "html.parser")
            plain_text = soup.get_text()
            # 使用正则表达式移除网址链接
            text = re.sub(r"http\S+", "", plain_text)

            return text

    @classmethod
    def _read_text(cls, file_path: str) -> str:
        """
        读取文本文件

        Args:
            file_path (str): 文件路径

        Returns:
            str: 文本文件内容
        """
        with open(file_path, "r", encoding = "utf-8") as file:
            return file.read()


class Documents:
    """
    获取已经分好类的 JSON 格式文档
    """
    
    def __init__(self, path: str = "") -> None:
        self.path = path

    def get_content(self):
        with open(self.path, mode = "r", encoding = "utf-8") as f:
            content = json.load(f)
        
        return content




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
