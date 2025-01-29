# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LLM.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-24
# * Version     : 1.0.092403
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union

import torch
from peft import PeftModel
from transformers import (
    AutoTokenizer, 
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# RAG
# ------------------------------
# 
PROMPT_TEMPLATE = {
    "RAG_PROMPT_TEMPLATE": """使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    "InternLM_PROMPT_TEMPLATE": """先对上下文进行内容总结，再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
}


class BaseModel:
    
    def __init__(self, path: str = "") -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str = "") -> str:
        pass
    
    def load_model(self):
        pass


class OpenAIChat(BaseModel):

    def __init__(self, path: str = "", model: str = "gpt-3.5-turbo-1106") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str = "") -> str:
        from openai import OpenAI
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")
        client.base_url = os.getenv("OPENAI_BASE_URL")
        
        history.append({
            "role": "user",
            "content": PROMPT_TEMPLATE["RAG_PROMPT_TEMPLATE"].format(question = prompt, context = content)
        })
        response = client.chat.completions.create(
            model = self.model,
            messages = history,
            max_tokens = 150,
            temperature = 0.1,
        )
        
        return response.choices[0].message.content


class InternLMChat(BaseModel):

    def __init__(self, path: str = "") -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history: List[dict], content: str = "") -> str:
        prompt = PROMPT_TEMPLATE["InternLM_PROMPT_TEMPLATE"].format(question = prompt, context = content)
        response, history = self.model.chat(
            self.tokenizer, 
            prompt, 
            history
        )

        return response
    
    def load_model(self): 
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, 
            trust_remote_code = True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, 
            torch_dtype = torch.float16, 
            trust_remote_code = True
        ).cuda()


class DashscopeChat(BaseModel):

    def __init__(self, path: str = "", model: str = "qwen-turbo") -> None:
        super().__init__(path)
        self.model= model

    def chat(self, prompt: str, history: List[dict], content: str = "") -> str:
        import dashscope
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        history.append({
            "role": "user",
            "content": PROMPT_TEMPLATE["RAG_PROMPT_TEMPLATE"].format(question = prompt, context = content)
        })
        response = dashscope.Generation.call(
            model = self.model,
            message = history,
            result_format = "message",
            max_tokens = 150,
            temperature = 0.1,
        )
        
        return response.output.choices[0].message.content


class ZhipuChat(BaseModel):

    def __init__(self, path: str = "", model: str = "glm-4") -> None:
        super().__init__(path) 
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str = "") -> str:
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key = os.getenv("ZHIPUAI_API_KEY"))
        
        history.append({
            "role": "user",
            "content": PROMPT_TEMPLATE["RAG_PROMPT_TEMPLATE"].format(question = prompt, context = content)
        })
        response = self.client.chat.completions.create(
            model = self.model,
            messages = history,
            max_tokens = 150,
            temperature = 0.1,
        )

        return response.choices[0].message


# ------------------------------
# Agent
# ------------------------------
class InternLM2Chat(BaseModel):
    
    def __init__(self, path: str = "") -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history: List[Dict], content: str = "", meta_instruction: str = "") -> str:
        """
        模型对话

        Args:
            prompt (str): 用户提示词
            history (List[Dict]): 消息历史
            content (str, optional): _description_. Defaults to "".
            meta_instruction (str, optional): 系统提示词. Defaults to "".

        Returns:
            str: _description_
        """
        response, history = self.model.chat(
            self.tokenizer, 
            prompt,
            history,
            temperature = 0.1,
            meta_instruction = meta_instruction,
        )
        
        return response, history 
    
    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print('================ Loading model ================')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, 
            trust_remote_code = True
        )
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.path, 
                torch_dtype = torch.float16, 
                trust_remote_code = True
            ).cuda().eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.path, 
                torch_dtype = torch.float16, 
                trust_remote_code = True
            ).eval()
        print('================ Model loaded ================')


# ------------------------------
# Eval
# ------------------------------
class BaseLLM:
    
    def __init__(self, path: str, model_name: str, adapter_path: str) -> None:
        self.path = path
        self.model_name = model_name
        self.adapter_path = adapter_path
    
    def build_chat(self, tokenizer, prompt, model_name):
        pass

    def load_model_and_tokenizer(self, path, model_name, device):
        pass

    def post_process(self, response, model_name):
        pass

    def get_pred(self, data: List, max_length: int, max_gen: int, prompt_format: str, device, out_path: str):
        pass


class internlm2Chat(BaseLLM):

    def __init__(self, path: str, model_name: str = "", adapter_path: str = "") -> None:
        super().__init__(path, model_name, adapter_path)

    def build_chat(self, tokenizer, prompt, model_name):
        prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def post_process(self, response, model_name):
        response = response.split("<|im_end|>")[0]

        return response

    def load_model_and_tokenizer(self, path, model_name, device, adapter_path):
        tokenizer = AutoTokenizer.from_pretrained(
            path, 
            trust_remote_code = True
        )
        model = AutoModelForCausalLM.from_pretrained(
            path, 
            trust_remote_code = True, 
            torch_dtype = torch.bfloat16
        ).to(device)
        if adapter_path:
            model = PeftModel.from_pretrained(model, model_id = adapter_path)
        # model eval
        model = model.eval()

        return model, tokenizer

    def get_pred(self, data: List, max_length: int, max_gen: int, prompt_format: str, device, out_path: str):
        model, tokenizer = self.load_model_and_tokenizer(self.path, device, self.adapter_path)
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            # 在中间截断，因为两头有关键信息
            tokenized_prompt = tokenizer(prompt, truncation = False, return_tensors = "pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens = True) + \
                         tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens = True)
            prompt = self.build_chat(prompt)
            input = tokenizer(prompt, truncation = False, return_tensors = "pt").to(device)
            context_length = input.input_ids.shape[-1]  # 表示喂进去的 tokens 的长度
            eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]

            output = model.generate(
                **input,
                max_new_tokens = max_gen,
                do_sample = False,
                temperature = 1.0,
                eos_token_id = eos_token_id,
            )[0]
            
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = self.post_process(pred)
            
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({
                    "pred": pred, 
                    "answers": json_obj["answers"], 
                    "all_classes": json_obj["all_classes"], 
                    "length": json_obj["length"]
                }, f, ensure_ascii = False)
                f.write('\n')


class Qwen2Chat(BaseLLM):
    
    def __init__(self, path: str, model_name: str = '', adapter_path: str = '') -> None:
        super().__init__(path, model_name, adapter_path) 
        
    def build_chat(self, prompt, instruct = None):
        if instruct is None:
            instruct = 'You are a helpful assistant.'
        prompt = f'<|im_start|>system\n{instruct}<im_end>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'

        return prompt

    def load_model_and_tokenizer(self, path, device, adapter_path):
        tokenizer = AutoTokenizer.from_pretrained(
            path, 
            trust_remote_code = True
        )
        model = AutoModelForCausalLM.from_pretrained(
            path, 
            trust_remote_code = True, 
            torch_dtype = torch.bfloat16
        ).to(device)
        # adapter_path = ''
        if adapter_path:
            model = PeftModel.from_pretrained(model, model_id = adapter_path)
            print(f"adapter loaded in {adapter_path}")
        
        model = model.eval()

        return model, tokenizer
    
    def get_pred(self, data, max_length, max_gen, prompt_format, device, out_path):
        model, tokenizer = self.load_model_and_tokenizer(self.path, device, self.adapter_path)
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            # 在中间截断,因为两头有关键信息.
            tokenized_prompt = tokenizer(prompt, truncation = False, return_tensors = "pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens = True) + \
                         tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens = True)

            prompts = self.build_chat(prompt, json_obj.get('instruction', None))
            inputs = tokenizer(prompts, truncation = False, return_tensors="pt").to(device)

            output = model.generate(
                inputs.input_ids,
                do_sample = True,
                temperature = 1.0,
                max_new_tokens = max_gen,
                top_p = 0.8,
            )
            
            pred = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output)]
            pred = tokenizer.batch_decode(pred, skip_special_tokens = True)[0]
            
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(
                    {
                        "pred": pred, 
                        "answers": json_obj["output"], 
                        "all_classes": json_obj.get("all_classes", None), 
                        "length": json_obj.get("length", None)
                    }, 
                    f, 
                    ensure_ascii = False
                )
                f.write('\n')
    



# 测试代码 main 函数
def main():
    model = InternLM2Chat("./downloaded_models/Shanghai_AI_Laboratory/internlm2-chat-20b")
    res = model.chat(prompt = "Hello", history = [], content = "", meta_instruction = "")
    print(res)

if __name__ == "__main__":
    main()
