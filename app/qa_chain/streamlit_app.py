# -*- coding: utf-8 -*-

# ***************************************************
# * File        : steamlit_app_demo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-04
# * Version     : 0.1.080417
# * Description : description
# * Link        : https://github.com/datawhalechina/llm-universe/blob/0ce94e828ce2fb63d47741098188544433c5e878/notebook/C4%20%E6%9E%84%E5%BB%BA%20RAG%20%E5%BA%94%E7%94%A8/streamlit_app.py
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from dotenv import load_dotenv, find_dotenv

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

from embedding_api.zhipuai_embedding import ZhipuAIEmbeddings

# 将父目录放入系统路径中
sys.path.append("../knowledge_lib") 
# 读取本地 .env 文件
_ = load_dotenv(find_dotenv())
# 载入 ***_API_KEY
os.environ["OPENAI_API_BASE"] = "https://api.chatgptid.net/v1"
zhipuai_api_key = os.environ["ZHIPUAI_API_KEY"]
# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def generate_response(input_text, openai_api_key):
     """
     定义一个函数，使用用户密钥对 OpenAI API 进行身份验证、发送提示并获取 AI 生成的响应。
     该函数接受用户的提示作为参数，并使用 st.info 来在蓝色框中显示 AI 生成的响应。
     """
     llm = ChatOpenAI(temperature = 0.7, openai_api_key = openai_api_key)
     output = llm.invoke(input_text)
     output_parser = StrOutputParser()
     output = output_parser.invoke(output)
     # st.info(output)
     return output


def get_vectordb():
     """
     函数返回持久化后的向量知识库
     """
     # 定义 Embeddings
     embedding = ZhipuAIEmbeddings()
     # 向量数据库持久化路径
     persist_directory = "data_base/vector_db/chroma"
     # 加载数据库
     vectordb = Chroma(
          persist_directory = persist_directory,
          embedding_function = embedding,
     )

     return vectordb


def get_chat_qa_chain(question: str, openai_api_key: str):
     """
     带有历史记录的问答链
     """
     vectordb = get_vectordb()
     llm = ChatOpenAI(
          model_name = "gpt-3.5-turbo", 
          temperature = 0, 
          openai_api_key = openai_api_key
     )
     memory = ConversationBufferMemory(
          memory_key = "chat_history",  # 与 prompt 的输入变量保持一致
          return_messages = True,  # 将消息列表的形式返回聊天记录，而不是单个字符串
     )
     retriever = vectordb.as_retriever()
     qa = ConversationBufferMemory.from_llm(
          llm, 
          retriever = retriever,
          memory = memory,
     )
     result = qa({
          "question": question
     })
     
     return result["answer"]


def get_qa_chain(question: str, openai_api_key: str):
     """
     不带历史记录的问答链
     """
     vectordb = get_vectordb()
     llm = ChatOpenAI(
          model = "gpt-3.5-turbo",
          temperature = 0,
          opanai_api_key = openai_api_key,
     )
     template = """"使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
     QA_CHAIN_PROMPT = PromptTemplate(
          input_variables = ["context", "question"],
          template = template,
     )
     qa_chain = RetrievalQA.from_chain_type(
          llm,
          retriever = vectordb.as_retriever(),
          return_source_documents = True,
          chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
     )
     result = qa_chain({"query": question})

     return result["result"]




# Streamlit 应用程序界面
def main():
     # 创建应用程序的标题
     st.title("🦜🔗 动手学大模型应用开发")
     # 添加一个文本输入框，供用户输入其 OpenAI API 密钥
     openai_api_key = st.sidebar.text_input("OpenAI API Key", type = "password")
     # 添加一个选择按钮来选择不同的模型
     # selected_method = st.sidebar.selectbox(
     #      "选择模式", 
     #      [
     #           "None", 
     #           "qa_chain", 
     #           "chat_qa_chain"
     #      ]
     # )
     selected_method = st.radio(
          "你想选择哪种模式进行对话？",
          [
               "None", 
               "qa_chain", 
               "chat_qa_chain"
          ],
          caption = [
               "不使用检索回答的普通模式", 
               "不带历史记录的检索问答模式", 
               "带历史记录的检索问答模式"
          ]
     )

     # 用于跟踪对话历史
     # 通过使用 st.session_state 来存储对话历史，
     # 可以在用户与应用程序交互时保留整个对话的上下文
     if "messages" not in st.session_state:
          st.session_state.messages = []
     
     # 当用户点击 Submit 时，generate_response 将使用用户的输入作为参数来调用该函数
     # with st.form("my_form"):
     #      text = st.text_area("Enter text:", "What are the three key pieces of advice for learning how to code?")
     #      submitted = st.form_submit_button("Submit")
     #      if not openai_api_key.startswith("sk-"):
     #           st.warning("Please enter your OpenAI API key!", icon = "")
     #      if submitted and openai_api_key.startswith("sk-"):
     #           generate_response(text, openai_api_key)
   
     messages = st.container(height = 300)
     if prompt := st.chat_input("Say something"):
          # 将用户输入添加到对话历史中
          st.session_state.messages.appen({
               "role": "user",
               "text": prompt,
          })
          # 调用 respond 函数获取回答
          if selected_method == "None":
               answer = generate_response(prompt, openai_api_key)
          if selected_method == "qa_chain":
               answer = get_qa_chain(prompt, openai_api_key)
          elif selected_method == "chat_qa_chain":
               answer = get_chat_qa_chain(prompt, openai_api_key)
          
          # 检查回答是否为 None
          if answer is not None:
               # 将 LLM 的回答添加到对话历史中
               st.session_state.messages.append({
                    "role": "assistant",
                    "text": answer,
               })
          
          # 显示整个对话历史
          for message in st.session_state.messages:
               if message["role"] == "user":
                    messages.chat_message("user").write(message["text"])
               else:
                    messages.chat_message("assistant").write(message["text"])

if __name__ == "__main__":
    main()
