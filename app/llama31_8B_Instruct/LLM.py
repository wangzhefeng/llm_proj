# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LLM.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-15
# * Version     : 0.1.081521
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
import warnings
warnings.filterwarnings("ignore")
from typing import Any, List, Optional

import torch
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class LLaMA3_1_LLM(LLM):
    """
    基于本地 llama3.1 自定义 LLM 类
    """
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_name_or_path: str):
        super(LLaMA3_1_LLM, self).__init__()
        
        print("正在从本地加载模型...")
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast = False,
            # trust_remote_code = True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype = torch.bfloat16,
            device_map = "auto",
            # trust_remote_code = True,
        )
        print("完成本地模型的加载")

    def _call(self, 
              prompt : str, 
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any
              ):
        # prompt template
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]
        # input ids
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            tokenize = False, 
            add_generation_prompt = True
        )
        # TODO
        model_inputs = self.tokenizer([input_ids], return_tensors = "pt").to(self.model.device)
        # TODO
        generated_ids = self.model.generate(
            model_inputs.input_ids, 
            max_new_tokens = 512,
        )
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens = True)[0]
        
        return response

    @property
    def _llm_type(self) -> str:
        return "LLaMA3_1_LLM" 



# 测试代码 main 函数
def main():
    # from LLM import LLaMA3_1_LLM

    llm = LLaMA3_1_LLM(
        model_name_or_path = "D:/projects/llms_proj/llm_proj/downloaded_models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    )
    print(llm("你好"))

if __name__ == "__main__":
    main()
