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

from Agent import Agent

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# params
llm_dir = "./downloaded_models/Shanghai_AI_Laboratory/internlm2-chat-7b"


agent = Agent(path = llm_dir)
print(agent.system_prompt)

response, _ = agent.text_completion(text = "你好", history = [])
print(response)

response, _ = agent.text_completion(text = "特朗普哪一年出生地？", history = _)
print(response)

response, _ = agent.text_completion(text = "周杰伦是谁？", history = _)
print(response)

response, _ = agent.text_completion(text = "书生浦语是什么？", history = _)
print(response)




# 测试代码 main 函数
def main():
    # result
    pass

if __name__ == "__main__":
    main()
