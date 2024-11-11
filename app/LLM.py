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


class LocalLLM(LLM):
    """
    基于本地下载模型自定义 LLM 类
    """
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_name_or_path: str, pad_token: bool = False) -> None:
        super(LocalLLM, self).__init__()
        print("正在从本地加载模型...")
        # 加载本地 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            use_fast = False,
            trust_remote_code = True,
        )
        if pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # print(f"tokenizer:\n{self.tokenizer}")
        
        # 加载本地 LLM 模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype = torch.bfloat16, 
            device_map = "auto",
            trust_remote_code = True,
        )
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path
        )
        # 开启梯度检查点
        self.model.enable_input_require_grads()
        # print(f"model:\n{self.model}")
        # print(f"model type:{self.model.dtype}")
        print("完成本地模型的加载")
    
    def _call(self,
              prompt: str, 
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        """
        模型推理
        """
        # prompt template
        messages = [
            {"role": "user", "content": prompt}
        ]
        # 模型输入
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        model_inputs = self.tokenizer([input_ids], return_tensors = "pt").to(self.model.device)
        generated_ids = self.model.generate(model_inputs.input_ids, attention_mask = model_inputs.attention_mask, max_new_tokens = 512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        # 模型输出
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens = True)

        return response[0]
    
    @property
    def _llm_type(self) -> str:
        return "LLM"


def get_tokenizer_model(model_path, pad_token: bool):
    """
    加载 tokenizer 和半精度模型
    """
    # 加载本地 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast = False,
        trust_remote_code = True
    )
    if pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"tokenizer:\n{tokenizer}")

    # 加载本地 LLM 模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,  
        torch_dtype = torch.bfloat16, 
        device_map = "auto",
        trust_remote_code = True,
    )
    # 开启梯度检查点
    model.enable_input_require_grads()
    print(f"model:\n{model}")
    print(f"model type:{model.dtype}")
    
    return tokenizer, model




# 测试代码 main 函数
def main():
    # from LLM import Qwen2_5_LLM

    llm = LocalLLM(
        model_name_or_path = "D:\projects\llms_proj\llm_proj\downloaded_models\qwen\Qwen2.5-7B-Instruct"
    )
    print(llm(prompt = "你是谁"))

if __name__ == "__main__":
    main()
