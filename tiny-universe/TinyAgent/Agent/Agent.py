# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Agent.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-09-24
# * Version     : 0.1.092414
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
import json5
from typing import Dict, List, Tuple, Optional, Union

from LLM import InternLM2Chat
from tool import Tools

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""


class Agent:
    """
    Agent 的结构是一个 React 的结构，提供一个 `system_prompt`，
    使得 LLM 知道自己可以调用哪些工具，并以什么样的格式输出。
    
    每次用户的提问，如要需要调用工具的话，都会进行两次的 LLM 调用：
        - 第一次解析用户的提问，选择调用的工具和参数；
        - 第二次将工具返回的结果与用户的提问整合，这样就可以实现一个 ReAct 的结构。
    """
    
    def __init__(self, path: str = "") -> None:
        # 工具
        self.tool = Tools()
        # system prompt
        self.system_prompt = self._build_system_input()
        self.model = InternLM2Chat(path)
    
    def _build_system_input(self):
        """
        构建 ReAct 形式的 system prompt
        该 system prompt 告诉 LLM，可以调用哪些工具，
        以什么样的方式输出，以及工具的描述信息和工具应该接受什么样的参数。
        """
        tool_descs = [TOOL_DESC.format(**tool) for tool in self.tool.toolConfig]
        tool_names = [tool["name_for_model"] for tool in self.tool.toolConfig]
        tool_descs = "\n\n".join(tool_descs)
        tool_names = ",".join(tool_names)
        sys_prompt = REACT_PROMPT.format(tool_descs = tool_descs, tool_names = tool_names)

        return sys_prompt

    def _parse_latest_plugin_call(self, text: str):
        """
        TODO
        """
        plugin_name, plugin_args = "", ""
        
        i = text.rfind("\nAction:")
        j = text.rfind("\nAction Input:")
        k = text.rfind("\nObservation:")
        if 0 <= i < j:  # if the text has "Action" and "Action input"
            if k < j:  # but does not contain "Observation"
                text = text.rstrip() + "\nObservation:"  # add it back
            k = text.rfind("\nObservation:")
            plugin_name = text[(i + len("\nAction")):j].strip()
            plugin_args = text[(j + len("\nAction Input:")):k].strip()
            text = text[:k]
        
        return plugin_name, plugin_args, text
    
    def _call_plugin(self, plugin_name, plugin_args):
        """
        TODO
        """
        plugin_args = json5.loads(plugin_args)
        if plugin_name == "google_search":
            return "\nObservation:" + self.tool.google_search(**plugin_args)
    
    def text_completion(self, text: str, history: List = []):
        """
        调用 LLM，根据 ReAct 的 Agent 的逻辑，调用 Tools 中的工具
        
        每次用户的提问，如要需要调用工具的话，都会进行两次的 LLM 调用：
            - 第一次解析用户的提问，选择调用的工具和参数；
            - 第二次将工具返回的结果与用户的提问整合，这样就可以实现一个 ReAct 的结构。

        Args:
            text (_type_): 用户的提问
            history (list, optional): 消息历史. Defaults to [].
        """
        # query
        text = f"\nQuestion: {text}"
        # 模型生成
        response, history = self.model.chat(
            prompt = text, 
            history = history, 
            meta_instruction = self.system_prompt
        )
        plugin_name, plugin_args, response = self._parse_latest_plugin_call(response)
        if plugin_name:
            response += self._call_plugin(plugin_name, plugin_args)
        # tool + person
        response, history = self.model.chat(
            prompt = response, 
            history = history, 
            meta_instruction = self.system_prompt
        )

        return response, history




# 测试代码 main 函数
def main():
    agent = Agent(path = "./downloaded_models/Shanghai_AI_Laboratory/internlm2-chat-20b")
    system_prompt = agent.system_prompt
    print(system_prompt)
    
    """
    ================ Loading model ================
    Loading checkpoint shards:   0%|          | 0/8 [00:00<?, ?it/s]
    Loading checkpoint shards:  12%|█▎        | 1/8 [00:01<00:09,  1.37s/it]
    Loading checkpoint shards:  25%|██▌       | 2/8 [00:03<00:12,  2.10s/it]
    Loading checkpoint shards:  38%|███▊      | 3/8 [00:08<00:15,  3.12s/it]
    Loading checkpoint shards:  50%|█████     | 4/8 [00:12<00:13,  3.47s/it]
    Loading checkpoint shards:  62%|██████▎   | 5/8 [00:17<00:12,  4.21s/it]
    Loading checkpoint shards:  75%|███████▌  | 6/8 [00:24<00:09,  4.94s/it]
    Loading checkpoint shards:  88%|████████▊ | 7/8 [00:30<00:05,  5.27s/it]
    Loading checkpoint shards: 100%|██████████| 8/8 [00:36<00:00,  5.57s/it]
    Loading checkpoint shards: 100%|██████████| 8/8 [00:36<00:00,  4.54s/it]
    ================ Model loaded ================
    Answer the following questions as best you can. You have access to the following tools:

    google_search: Call this tool to interact with the 谷歌搜索 API. What is the 谷歌搜索 API useful for? 谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。 Parameters: [{'name': 'search_query', 'description': '搜索关键词或短语', 'required': True, 'schema': {'type': 'string'}}] Format the arguments as a JSON object.

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [google_search]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!
    """

if __name__ == "__main__":
    main()
