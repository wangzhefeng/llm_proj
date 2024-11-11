# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LLM.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-10
# * Version     : 0.1.111013
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
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    LlamaTokenizerFast,
)
from langchain.callbacks.manager import CallbackManagerForLLMRun


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Qwen2_5_LLM(LLM):
    """
    基于本地 Qwen2.5 自定义 LLM 类
    """
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_name_or_path: str) -> None:
        super(Qwen2_5_LLM, self).__init__()

        print("正在从本地加载模型...")
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            use_fast = False,
        )
        # model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype = torch.bfloat16, 
            device_map = "auto"
        )
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path
        )
        print("完成本地模型的加载")
    
    def _call(self,
              prompt: str, 
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # prompt template
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        
        # TODO
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        # print(f"input_ids:\n{input_ids}")
        
        # TODO
        model_inputs = self.tokenizer([input_ids], return_tensors = "pt").to(self.model.device)
        # print(f"model_inputs:\n{model_inputs}")
        # print(f"model_inputs.input_ids:\n{model_inputs.input_ids}")
        
        # 生成文本
        generated_ids = self.model.generate(
            model_inputs.input_ids, 
            attention_mask = model_inputs.attention_mask, 
            max_new_tokens = 512
        )
        # print(f"generated_ids:\n{generated_ids}")
        # print(f"generated_ids length:\n{len(generated_ids)}")
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # print(f"generated_ids:\n{generated_ids}")
        # print(f"generated_ids length:\n{len(generated_ids)}")
        
        # result
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens = True)
        # print(f"response:\n{response}")
        # print(f"response length: {len(response[0])}")

        return response[0]
    
    @property
    def _llm_type(self) -> str:
        return "Qwen_2_5_LLM"




# 测试代码 main 函数
def main():
    # from LLM import Qwen2_5_LLM

    llm = Qwen2_5_LLM(
        model_name_or_path = "D:\projects\llms_proj\llm_proj\downloaded_models\qwen\Qwen2.5-7B-Instruct"
    )
    print(llm("你是谁"))

if __name__ == "__main__":
    main()
