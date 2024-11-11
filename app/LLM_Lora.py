# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LLM_Lora.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-11
# * Version     : 0.1.111116
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

import pandas as pd
import torch
from datasets import Dataset
from langchain.llms.base import LLM
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    GenerationConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
# from swanlab.integration.transformers import SwanLabCallback
# import swanlab

from LLM import LocalLLM, get_tokenizer_model

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}.")


# ------------------------------
# 微调数据加载和预处理
# ------------------------------
def __process_func(example, tokenizer, model_mode: str, max_length: int = 384, full_p_tuning: bool = False):
    """
    微调数据格式化
    """
    # 分词器会将一个中文字切分为多个 token，因此需要放开一些最大长度，保证数据的完整性
    MAX_LENGTH = max_length
    # 指令集构建
    if model_mode == "llama":
        # llama lora
        instruction_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{example['instruction']}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        response_prompt = f"{example["output"]}<|eot_id|>"
    elif model_mode == "qwen" and not full_p_tuning:
        # qwen 2.5 lora
        instruction_prompt = f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
        # qwen 2.5 lora swan
        instruction_prompt = f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
        response_prompt = f"{example['output']}"
    elif model_mode == "qwen" and full_p_tuning:
        # qwen P-tuning/full fine-tuning
        instruction_prompt = f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
        response_prompt = f"<|im_start|>assistant\n{example["output"]}<|im_end|>\n"
    instruction = tokenizer(instruction_prompt, add_special_tokens = False)  # 不在开头加 special_tokens
    response = tokenizer(response_prompt, add_special_tokens = False)
    # input_ids/attention_mask/labels
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为 eos token 也是要关注的, 所以补充为 1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    # output
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    
    return out


def get_tuning_data(tuning_data_path):
    """
    加载和预处理微调数据集
    """
    # 微调数据加载
    tuning_df = pd.read_json(tuning_data_path)
    tuning_ds = Dataset.from_pandas(tuning_df)
    print(tuning_ds[:3])

    # 数据格式化处理
    tokenized_id = tuning_ds.map(__process_func, remove_columns = tuning_ds.column_names)
    print(tokenized_id)
    # print(tokenizer.decode(tokenized_id[0]["input_ids"]))
    # print(tokenizer.decode(filter(lambda x: x != -100, tokenized_id[1]["labels"])))
    
    return tokenized_id


# TODO
def get_tuning_data(data_path):
    """
    微调数据加载和预处理
    """
    # ------------------------------
    # 微调输入数据和输出数据
    # ------------------------------
    input_file = os.path.join(data_path, "DISC-Law-SFT-Pair-QA-released.jsonl")
    output_file = os.path.join(data_path, "DISC-Law-SFT-Pair-QA-released-new.jsonl")
    # ------------------------------
    # 微调数据预处理
    # ------------------------------
    # 定义固定的 Instruction
    INSTRUCTION = "你是一个法律专家，请根据用户的问题给出专业的回答"
    # 下载数据预处理
    with open(input_file, "r", encoding = "utf-8") as infile, \
        open(output_file, "w", encoding = "utf-8") as outfile:
        for line in infile:
            # 读取每一行并解析 JSON
            data = json.loads(line)
            # 创建新的字典，包含 instruction,input,output
            new_data = {
                "instruction": INSTRUCTION,
                "input": data["input"],
                "output": data["output"],
            }
            # 将新的字典写入输出文件
            json.dump(new_data, outfile, ensure_ascii=False)
            outfile.write('\n')
    # log
    print(f"处理完成。输出文件：{output_file}")
    # ------------------------------
    # 微调训练数据和测试数据
    # ------------------------------
    # 微调训练数据加载
    train_df = pd.read_json(outfile)[5:5000]
    test_df = pd.read_json(outfile)[:5]
    train_ds = Dataset.from_pandas(train_df)
    # 数据格式化处理
    train_dataset = train_ds.map(__process_func, remove_columns = train_ds.column_names)

    return train_dataset, test_df


def model_training(model, tokenizer, 
                      train_dataset, train_epochs = 3,
                      lora_path: str = None, lora_r: int = 8, 
                      lora_alpha: int = 32, lora_dropout: float = 0.1):
    """
    LoRA 微调
    """
    # 定义 LoraConfig
    lora_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM,
        target_modules = [
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", 
            "up_proj", "down_proj",
        ],
        inference_mode = False,  # 训练模式
        r = lora_r,  # Lora 秩
        lora_alpha = lora_alpha,  # Lora alpha
        lora_dropout = lora_dropout,  # dropout 比例
    )
    print(lora_config)

    # 创建 Peft 模型
    peft_model = get_peft_model(model = model, peft_config = lora_config)
    print(peft_model.print_trainable_parameters())

    # 配置 LoRA 训练参数
    args = TrainingArguments(
        output_dir = lora_path,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        logging_steps = 10,
        num_train_epochs = train_epochs,
        save_steps = 100,  # 快速演示设置 10，建议设置为 100
        learning_rate = 1e-4,
        save_on_each_node = True,
        gradient_checkpointing = True,
        report_to = "none",
    )
    # 使用 Trainer 训练
    trainer = Trainer(
        model = peft_model,
        args = args,
        train_dataset = train_dataset,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, padding = True),
        callbacks = [],
    )
    trainer.train()
    
    return lora_config


def model_inference(model, tokenizer, lora_path, lora_config, messages):
    """
    加载 LoRA 权重推理
    """ 
    # 加载 loRA 权重
    tuned_model = PeftModel.from_pretrained(model, model_id = lora_path, config = lora_config)
    # 模型输入
    text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    model_inputs = tokenizer([text], return_tensors = "pt").to(device)
    generated_ids = tuned_model.generate(model_inputs.input_ids, max_new_tokens = 512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    # 模型输出
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)[0]
    
    return response




# 测试代码 main 函数
def main():
    # ------------------------------
    # 数据、模型、参数地址
    # ------------------------------
    # 微调训练数据地址
    tuning_data_path = "D:\projects\llms_proj\llm_proj\dataset\DISC-Law"
    # 模型地址
    model_path = "D:\projects\llms_proj\llm_proj\downloaded_models\qwen\Qwen2.5-7B-Instruct"
    # LoRA 输出对应 checkpoint 地址
    lora_path = 'D:\projects\llms_proj\llm_proj\\app\output\qwen2_5_7B_instruct_lora' 
    # ------------------------------
    # 微调数据加载和预处理
    # ------------------------------
    train_dataset, test_df = get_tuning_data(data_path = tuning_data_path)
    # ------------------------------
    # 加载 tokenizer 和半精度模型
    # ------------------------------
    tokenizer, model = get_tokenizer_model(model_path = model_path)
    # ------------------------------
    # 模型微调
    # ------------------------------
    model_training(modoe = model, tokenizer = tokenizer, train_dataset = train_dataset, test_df = test_df, lora_path = lora_path)

    # 构建 prompt template
    prompt = "你是谁？"
    messages = [
        {
            "role": "system",
            "content": "现在你要扮演皇帝身边的女人--甄嬛",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

if __name__ == "__main__":
    main()
