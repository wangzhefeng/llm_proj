# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-26
# * Version     : 0.1.032602
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import gradio as gr


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def greet(name):
    return "Hello " + name + "!"


demo = gr.Interface(fn = greet, inputs = "text", outputs = "text")
demo.launch()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
