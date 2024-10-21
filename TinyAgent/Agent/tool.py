# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tool.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-24
# * Version     : 0.1.092414
# * Description : 1.首先，要在 tools 中添加工具的描述信息；
# *               2.然后，在 tools 中添加工具的具体实现；
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import json
import requests
from dotenv import find_dotenv, load_dotenv

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 读取本地/项目环境变量
_ = load_dotenv(find_dotenv())
GOOGLE_SERPER_API_KEY = os.environ["GOOGLE_SERPER_API_KEY"]


class Tools:
    """
    构造一些工具，比如 Google Search。添加一些工具的描述信息和具体的实现方式。
    """
    
    def __init__(self) -> None:
        self.toolConfig = self._tools()

    def _tools(self):
        """
        添加工具的描述信息，是为了在构造 `system_prompt` 的时候，
        让模型能够知道可以调用哪些工具，以及工具的描述信息和参数。
        """
        tools = [
            {
                "name_for_human": "谷歌搜索",
                "name_for_model": "google_search",
                "description_for_model": "谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。",
                "parameters": [
                    {
                        "name": "search_query",
                        "description": "搜索关键词或短语",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                ],
            },
        ]
        
        return tools

    def google_search(self, search_query: str):
        """
        使用 Google 搜索功能的话需要去 serper 官网申请一下 token: https://serper.dev/dashboard，
        然后在 `X-API-KEY` 填写 key，这个 key 每人可以免费申请一个，且有 2500 次的免费调用额度，足够做实验用

        Args:
            search_query (str): 查询内容/问题

        Returns:
            _type_: 查询结果
        """
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": search_query})
        headers = {
            "X-API-KEY": GOOGLE_SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        # request
        response = requests.request(
            method = "POST", 
            url = url, 
            headers = headers, 
            data = payload,
        ).json()

        return response["organic"][0]["snippet"]




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
