# -*- coding: utf-8 -*-

# ***************************************************
# * File        : chatBot.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-17
# * Version     : 0.1.081717
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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# 在侧边栏中创建一个标题和一个链接
with st.sidebar:
    st.markdown("## LlaMA3.1 LLM")
    "[开源大模型指南 self-llm](https://github.com/datawhalechina/self-llm.git)"


# 创建一个标题和一个副标题
st.title("💬 LLaMA3.1 Chatbot")
st.caption("🚀 A streamlit chatbot powered by Self-LLM")


# 定义一个函数，用于获取模型和 tokenizer
@st.cache_resource
def get_model():
    # 模型路径
    model_name_or_path = "D:\projects\llms_proj\llm_proj\downloaded_models\LLM-Research\Meta-Llama-3.1-8B-Instruct"
    # ------------------------------
    # 加载本地 LlaMA-3.1-8B-Instruct 模型
    # ------------------------------
    # 加载 LlaMA-3.1-8B-Instruct tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        # use_fast = False,
        trust_remote_code = True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 加载本地 LlaMA3.1-8B-Instruct 模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,  
        torch_dtype = torch.bfloat16, 
        # device_map = "auto",
        # trust_remote_code = True,
    ).cuda()

    return tokenizer, model


# 加载 LlaMA3.1 的模型和 tokenizer
tokenizer, model = get_model()


# 如果 session_state 中没有 "messages"，则创建一个包含默认消息的列表
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 遍历 session_state 中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)
    # 将用户输入添加到 session_state 中的 messages 列表中
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 将对话输入模型，获得返回
    input_ids = tokenizer.apply_chat_template(st.session_state["messages"],tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
    generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # 将模型的输出添加到 session_state 中的 messages列表中
    st.session_state.messages.append({"role": "assistant", "content": response})
    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)
    print(st.session_state)



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
