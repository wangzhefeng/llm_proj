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

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Tools:
    
    def __init__(self) -> None:
        self.toolConfig = self._tools()
    
    def _tools(self):
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
                    }
                ],
            }
        ]
         
        return tools

    def google_search(self, search_query: str):
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": search_query})
        headers = {
            "X-API-KEY": "修改为自己的 key",
            "Content-Type": "application/json",
        }
        response = requests.request(
            "POST", 
            url, 
            headers = headers, 
            data = payload,
        ).json()

        return response["organic"][0]["snippet"]


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
