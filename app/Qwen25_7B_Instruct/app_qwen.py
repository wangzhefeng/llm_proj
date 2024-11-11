# -*- coding: utf-8 -*-

# ***************************************************
# * File        : app_qwen.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-11
# * Version     : 0.1.111120
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
import json
import time

from dotenv import find_dotenv, load_dotenv
import streamlit as st
from openai import OpenAI

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
_ = load_dotenv(find_dotenv())
# x.ai api key
XAI_API_KEY = os.getenv("XAI_API_KEY")


# openai client
client = OpenAI(
    api_key = XAI_API_KEY,
    base_url = "https://api.x.ai/v1",
    # base_url = "http://localhost:8000/v1",
)


def make_api_call(messages, max_tokens, is_final_answer = False):
    """
    模型调用

    Args:
        messages (_type_): _description_
        max_tokens (_type_): _description_
        is_final_answer (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # 三次尝试机会
    for attempt in range(3):
        try:
            # LLM 生成内容
            response = client.chat.completions.create(
                model = "Qwen2.5-7B-Instruct",
                messages = messages,
                max_tokens = max_tokens,
                temperature = 0.2,
                response_format = {"type": "json_object"},
            )
            content = response.choices[0].message.content
            print(f"Raw API response: {content}")
            # 结果内容解析
            try:
                return json.load(content)
            except json.JSONDecodeError as json_error:
                print(f"JSON 解析错误: {json_error}")
                return {
                    "title": "API Response",
                    "content": content,
                    "next_action": "final_answer" if is_final_answer else "continue"
                }
        except Exception as e:
            if attempt == 2:
                return {
                    "title": "Error",
                    "content": f"Faild after 3 attempts. Error: {str(e)}",
                    "next_action": "final_answer",
                }
            # 重试前等待 1s
            time.sleep(1)


def generate_response(prompt):
    """
    生成推理内容
    """
    # messages: prompt template
    messages = [
        {
            "role": "system", "content": """
            你是一位具有高级推理能力的专家。你的任务是提供详细的、逐步的思维过程解释。对于每一步：
            1. 提供一个清晰、简洁的标题，描述当前的推理阶段。
            2. 在内容部分详细阐述你的思维过程。
            3. 决定是继续推理还是提供最终答案。

            输出格式说明：
            输出请严格遵循 JSON 格式, 包含以下键：'title'，'content'，'next_action'(值只能为 'continue' 或 'final_answer' 二者之一)。

            关键指示:
            - 至少使用 5 个不同的推理步骤。
            - 承认你作为 AI 的局限性，明确说明你能做什么和不能做什么。
            - 主动探索和评估替代答案或方法。
            - 批判性地评估你自己的推理；识别潜在的缺陷或偏见。
            - 当重新审视时，采用根本不同的方法或视角。
            - 至少使用 3 种不同的方法来得出或验证你的答案。
            - 在你的推理中融入相关的领域知识和最佳实践。
            - 在适用的情况下，量化每个步骤和最终结论的确定性水平。
            - 考虑你推理中可能存在的边缘情况或例外。
            - 为排除替代假设提供清晰的理由。

            示例 JSON 输出：
            {
                "title": "初步问题分析",
                "content": "为了有效地解决这个问题，我首先会将给定的信息分解为关键组成部分。这涉及到识别...[详细解释]...通过这样构建问题，我们可以系统地解决每个方面。",
                "next_action": "continue"
            }

            记住：全面性和清晰度至关重要。每一步都应该为最终解决方案提供有意义的进展。
            再次提醒：输出请务必严格遵循 JSON 格式，包含以下键：'title'，'content'，'next_action'(值只能为 'continue' 或 'final_answer' 二者之一)。
            """
        },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "现在我将一步步思考，从分析问题开始并将问题分解。"}
    ]
    # ------------------------------
    # 推理过程
    # ------------------------------
    # 记录每次推理信息
    steps = []
    # 推理次数
    step_count = 1
    # 推理总思考时间
    total_thinking_time = 0
    while True:
        # 模型调用，生成内容
        start_time = time.time()
        step_data = make_api_call(messages = messages, max_tokens = 1000)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        # 生成结果解析
        title = step_data.get('title', f'Step {step_count}')
        content = step_data.get('content', 'No content provided')
        next_action = step_data.get('next_action', 'continue')
        # 记录每次推理信息
        steps.append((f"Step {step_count}: {title}", content, thinking_time))
        # 更新 messages
        messages.append({"role": "assistant", "content": json.dumps(step_data)}) 
        # 最多 25 步，以防止无限的思考，可以适当调整。
        if next_action == 'final_answer' or step_count > 25: 
            break
        # 更新迭代次数
        step_count += 1
        # 在结束时生成总时间
        yield steps, None
    # ------------------------------
    # 生成最终答案
    # ------------------------------
    # messages: prompt template
    messages.append({"role": "user", "content": "请根据你上面的推理提供最终答案。"})
    # 模型调用，生成内容
    start_time = time.time()
    final_data = make_api_call(messages = messages, max_tokens = 1000, is_final_answer = True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    # 生成结果解析
    final_content = final_data.get('content', '没有推理出最终结果')
    steps.append(("最终推理结果", final_content, thinking_time))

    yield steps, total_thinking_time




# 测试代码 main 函数
def main():
    st.set_page_config(
        page_title = "Qwen2.5 o1-like Reasoning Chain", 
        page_icon = "💬",
        layout = "wide",
    )
    st.title("Qwen2.5 实现类似 o1 model 的推理链")
    st.caption("🚀 A streamlit implementation")
    st.markdown("通过 vLLM 部署调用 Qwen2.5-7B-Instruct 并实现类似 OpenAI o1 model 的长推理链效果以提高对复杂问题的推理准确性。")
    # 用户输入查询
    user_query = st.text_input("输入问题：", placeholder = "示例：strawberry 中有多少个字母 r？")
    if user_query:
        st.write("正在生成推理链中...")
        # 创建空元素以保存生成的文本和总时间
        response_container = st.empty()
        time_container = st.empty()
        # 生成并显示响应
        for steps, total_thinking_time in generate_response(user_query):
            with response_container.container():
                for i, (title, content, thinking_time) in enumerate(steps):
                    if title.startswith("最终推理结果"):
                        st.markdown(f"### {title}")
                        st.markdown(content.replace("\n", "<br>"), unsafe_allow_html = True)
                    else:
                        with st.expander(title, expanded = True):
                            st.markdown(content.replace("\n", "<br>"), unsafe_allow_html = True)
            # 仅在结束时显示总时间
            if total_thinking_time is not None:
                time_container.markdown(f"**总推理时间：{total_thinking_time:.2f} 秒**")

if __name__ == "__main__":
    main()
