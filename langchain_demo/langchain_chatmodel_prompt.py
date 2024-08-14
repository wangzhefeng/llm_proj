# -*- coding: utf-8 -*-

# ***************************************************
# * File        : langchain_chatmodel_prompt.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-07-14
# * Version     : 0.1.071415
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]



# 定义对话系统预设消息模版
template = "你是一个翻译助手，可以将{input_language}翻译为{output_language}。"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# 定义用户消息模版
human_template = "{talk}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 构建聊天提示模版
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt,
])

# 生成聊天消息
messages = chat_prompt.format_prompt(
    input_language = "中文",
    output_language = "英文",
    talk = "我爱编程",
)

# 打印生成的聊天消息
for message in messages:
    print(message)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
