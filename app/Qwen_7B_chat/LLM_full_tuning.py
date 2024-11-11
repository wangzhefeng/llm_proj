# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LLM_full_tuning.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-10
# * Version     : 0.1.111018
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

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq, 
    TrainingArguments, 
    HfArgumentParser, 
    Trainer
)
from dataclasses import dataclass, field
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


@dataclass
class FinetuneArguments:
    # 微调参数
    # field：dataclass 函数，用于指定变量初始化
    model_path: str = field(default = "../../model/qwen/Qwen-7B-Chat/")

 
def process_func(example, tokenizer):
    """
    用于处理数据集的函数
    """
    # 分词器会将一个中文字切分为多个 token，因此需要放开一些最大长度，保证数据的完整性
    MAX_LENGTH = 128 

    # 指令集构建
    instruction = tokenizer(
        "\n".join(["<|im_start|>system", "现在你要扮演皇帝身边的女人--甄嬛.<|im_end|>" + "\n<|im_start|>user\n" + example["instruction"] + example["input"] + "<|im_end|>\n"]).strip(), 
        add_special_tokens = False,  # 在开头加 special_tokens
    )
    response = tokenizer(
        "<|im_start|>assistant\n" + example["output"] + "<|im_end|>\n", 
        add_special_tokens = False
    )

    # input_ids/attention_mask/labels
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为 eos token 也是要关注的所以补充为 1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  # Qwen 的特殊构造就是这样的

    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    } 




# 测试代码 main 函数
def main():
    # 解析参数
    # Parse 命令行参数
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # 处理数据集
    # 将 JSON 文件转换为 CSV 文件
    df = pd.read_json('./data/huanhuan.json')
    ds = Dataset.from_pandas(df) 

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        finetune_args.model_path, 
        use_fast = False, 
        trust_remote_code = True
    )
    tokenizer.pad_token_id = tokenizer.eod_id
    
    # 将数据集变化为 token 形式
    tokenized_id = ds.map(process_func, remove_columns = ds.column_names)
    
    # 创建模型并以半精度形式加载
    model = AutoModelForCausalLM.from_pretrained(
        finetune_args.model_path, 
        trust_remote_code = True, 
        torch_dtype = torch.half, 
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    )
    
    # 使用 Trainer 训练
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_id,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, padding = True),
    )
    trainer.train()
    
    # 模型推理
    response, history = model.chat(
        tokenizer, 
        "你是谁", 
        history = [], 
        system = "现在你要扮演皇帝身边的女人--甄嬛."
    )
    print(response)

if __name__ == "__main__":
    main()
