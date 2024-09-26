# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-24
# * Version     : 1.0.092404
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

from tinyAgent.Agent import Agent

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

# TODO
agent = Agent("./downloaded_models/internlm2-chat-20b")
print(agent.system_prompt)

# TODO
response, _ = agent.text_completion(text = "你好", history = [])
print(response)

# TODO
response, _ = agent.text_completion(text = "特朗普哪一年出生地？", history = _)
print(response)

# TODO
response, _ = agent.text_completion(text = "周杰伦是谁？", history = _)
print(response)

# TODO
response, _ = agent.text_completion(text = "书生浦语是什么？", history = _)
print(response)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
