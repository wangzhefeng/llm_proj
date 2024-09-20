# -*- coding: utf-8 -*-

# ***************************************************
# * File        : langchain_udf_prompt.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-07-11
# * Version     : 0.1.071123
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

from langchain.prompts import StringPromptTemplate
from langchain.pydantic_v1 import BaseModel, validator

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


delimiter = "####"
PROMPT = f"""将每个用户的信息用{delimiter}字符分割，并按照 JSON 格式提取姓名、职业和爱好信息。
示例如下：\n"""


class PersonInfoPromptTemplate(StringPromptTemplate, BaseModel):
    """
    自定义提示模板，用于生成关于人物信息的 JSON 格式输出
    """

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """
        验证输入变量
        """
        if "name" not in v:
            raise ValueError("name 字段必须包含在 input_variable 中。")
        if "occupation" not in v:
            raise ValueError("occupation 字段必须包含在 input_variable 中。")
        if "fun_fact" not in v:
            raise ValueError("fun_fact 字段必须包含在 input_variable 中。")
        
        return v
    
    def format(self, **kwargs) -> str:
        """
        格式化输入，生成 JSON 格式输出
        """
        person_info = {
            "name": kwargs.get("name"),
            "occupation": kwargs.get("occupation"),
            "fun_fact": kwargs.get("fun_fact"),
        }

        return PROMPT + json.dumps(person_info, ensure_ascii = False)

    def _prompt_type(self):
        """
        指定模板类型
        """
        return "person-info"
    

# 使用模板
person_info_template = PersonInfoPromptTemplate(input_variables = ["name", "occupation", "fun_fact"])
prompt_output = person_info_template.format(
    name = "张三 ",
    occupation = "软件工程师 ",
    fun_fact = "喜欢攀岩"
)

print(prompt_output)
 





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
