# -*- coding: utf-8 -*-

# ***************************************************
# * File        : langchain_outputparse2.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-07-14
# * Version     : 0.1.071418
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
import re
import json
from typing import Type

from langchain.schema import BaseOutputParser
from langchain.pydantic_v1 import BaseModel, ValidationError, Field

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class CustomOutputParser(BaseOutputParser[BaseModel]):
    
    pydantic_object: Type[T]

    def parse(self, text: str) -> BaseModel:
        """
        解析文本到 Pydantic 模型

        Args:
            text (str): 要解析的文本

        Returns:
            BaseModel: Pydantic 模型的一个实例
        """
        try:
            # 贪婪搜索第一个 JSON 候选
            match = re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
            json_str = match.group() if match else ""
            json_object = json.loads(json_str, strict = False)
            return self.pydantic_object.parse_obj(json_object)
        except (json.JSONDecodeError, ValidationError) as e:
            name = self.pydantic_object.__name__
    
    def get_format_instructions(self) -> str:
        """
        获取格式说明
        Returns:
            格式说明的字符串
        """
        schema = self.pydantic_object.schema()
        # 移除不必要的字段
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # 确保 json 在上下文中格式正确（使用双引号）
        schema_str = json.dumps(reduced_schema)

        return CUSTOM_FORMAT_INSTRUCTIONS.format(schema = schema_str)

    @property
    def _type(self) -> str:
        """
        获取解析器类型
        Returns:
            str: 解析器的类型字符串
        """
        return "custom output parser"


class ExpenseRecord(BaseModel):
    
    amount: float = Field(description = "花费金额")
    category: str = Field(description = "花费类别")
    date: str = Field(description = "花费日期")
    description: str = Field(description = "花费描述")

    # 创建 Pydantic 输出解析器实例
    parser = CustomOutputParser(pydantic_object = ExpenseRecord)
    # 定义获取花费记录的提示模板
    expense_template = """
    请将这些花费记录在我的账单中。
    我的花费记录是：{query}
    格式说明：
    {format_instructions}
    """
    # 使用提示模板创建实例
    prompt = PromptTemplate(
        template = expense_template,
        input_variables = ["query"],
        partial_variables = {
            "format_instructions": parser.get_format_instructions()
        },
    )
    # 格式化提示词
    _input = prompt.format_prompt(query = "昨天白天我去超市花了 45 元买日用品，晚上我又花了 20 元打车。")
    # 创建 OpenAI 模型实例
    model = OpenAI(model_name = "text_davinci-003", temperature = 0)
    # 使用模型处理格式化后的提示词
    output = model(_input.to_string())
    # 解析输出结果
    expense_record = parser.parse(output)
    # 遍历并打印花费记录的各个参数
    for parameter in expense_record.__field__:
        print(f"{parameter}: {expense_record.__dict__[parameter]},
                             {type(expense_record.__dict__[parameter])}")





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
