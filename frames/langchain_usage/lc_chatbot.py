# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lc_chatbot.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-16
# * Version     : 0.1.111601
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
from typing import Sequence
from typing_extensions import Annotated, TypedDict

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, trim_messages, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
_ = load_dotenv(find_dotenv())
# notebook
# import getpass
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["OPENAI_API_KEY"] = getpass.getpass()
# script
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ------------------------------
# llm
# ------------------------------
model = ChatOpenAI(model = "gpt-3.5-turbo")

# ------------------------------
# prompt templates
# ------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ------------------------------
# managing conversation history 
# ------------------------------
trimmer = trim_messages(
    max_tokens = 65,
    strategy = "last",
    token_counter = model,
    include_system = True,
    allow_partial = False,
    start_on = "human",
)

# messages = [
#     SystemMessage(content="you're a good assistant"),
#     HumanMessage(content="hi! I'm bob"),
#     AIMessage(content="hi!"),
#     HumanMessage(content="I like vanilla ice cream"),
#     AIMessage(content="nice"),
#     HumanMessage(content="whats 2 + 2"),
#     AIMessage(content="4"),
#     HumanMessage(content="thanks"),
#     AIMessage(content="no problem!"),
#     HumanMessage(content="having fun?"),
#     AIMessage(content="yes!"),
# ]
# trimmer.invoke(messages)


# ------------------------------
# message persistence
# ------------------------------
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


# 定义一个调用模型的函数
def call_model(state: State):
    chain = prompt | model
    trimmed_messages = trimmer.invoke(state["messages"])
    response = chain.invoke({
        "messages": trimmed_messages, 
        "language": state["language"]
    })
 
    return {"messages": response}


# 定义一个新 graph
workflow = StateGraph(state_schema=State)
# 定义 graph 的单个节点
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
# 增加记忆
app = workflow.compile(checkpointer=MemorySaver())


# ------------------------------
# streaming
# ------------------------------
config = {"configurable": {"thread_id": "abc789"}}
query = "Hi I'm Todd, please tell me a joke."
language = "English"

input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):
        print(chunk.content, end = "|")




# 测试代码 main 函数
def main():
    from langchain_core.messages import HumanMessage, AIMessage

    # 模型调用
    res = model.invoke(
        [
            HumanMessage(content = "Hi! I'm Bob"),
            AIMessage(content = "Hello Bob! How can I assist you today?"),
            HumanMessage(content = "What's my name?")
        ]
    )
    
    # 对话
    # 创建一个每次传递到可运行程序的 config
    config = {"configurable": {"thread_id": "abc123",}}
    query = "Hi! I'm Bob."
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    # 继续对话
    query = "What's my name?"
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    # 更改配置引用不同的 thread_id，重新开始对话
    config = {"configurable": {"thread_id": "abc234"}}
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    # 回到原来的对话
    config = {"configurable": {"thread_id": "abc123"}}
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    
    # 对话
    config = {"configurable": {"thread_id": "abc345"}}
    query = "Hi! I'm Jim."
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    
    query = "What is my name?"
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    
    # 对话
    config = {"configurable": {"thread_id": "abc456"}}
    query = "Hi! I'm Bob."
    language = "Spanish"

    input_messages = [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    output["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()
