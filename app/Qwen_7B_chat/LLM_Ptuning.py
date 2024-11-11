# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LLM_Ptuning.py
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
    Trainer
)
from peft import (
    PromptEncoderConfig, 
    TaskType, 
    get_peft_model, 
    PromptEncoderReparameterizationType
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def process_func(example, tokenizer):
    """
    指令数据集预处理
    """
    # 分词器会将一个中文字切分为多个 token，因此需要放开一些最大长度，保证数据的完整性
    MAX_LENGTH = 384
    # 指令集构建
    instruction = tokenizer(
        "\n".join(["<|im_start|>system", "现在你要扮演皇帝身边的女人--甄嬛.<|im_end|>" + "\n<|im_start|>user\n" + example["instruction"] + example["input"] + "<|im_end|>\n"]).strip(),
        add_special_tokens = False,  # 不在开头加 special_tokens
    )
    response = tokenizer(
        "<|im_start|>assistant\n" + example["output"] + "<|im_end|>\n", 
        add_special_tokens = False
    )
    # input_ids/attention_mask/labels
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为 eos token 也是要关注的所以 补充为 1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  # Qwen 的特殊构造就是这样的
    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    # output
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

    return out


# loraConfig
config = PromptEncoderConfig(
    task_type = TaskType.CAUSAL_LM, 
    num_virtual_tokens = 10,
    encoder_reparameterization_type = PromptEncoderReparameterizationType.MLP,
    encoder_dropout = 0.1, 
    encoder_num_layers = 5, 
    encoder_hidden_size = 1024
)

# 配置训练参数
args = TrainingArguments(
    output_dir = "./output/Qwen",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 2,
    logging_steps = 10,
    num_train_epochs = 3,
    # gradient_checkpointing=True,
    save_steps = 100,
    learning_rate = 1e-4,
    save_on_each_node = True
)




# 测试代码 main 函数
def main():
    model_path = ""

    # 处理数据集，并将 JSON 文件转换为 CSV 文件
    df = pd.read_json('./dataset/huanhuan.json')
    ds = Dataset.from_pandas(df)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast = False, 
        trust_remote_code = True
    )
    tokenizer.pad_token_id = tokenizer.eod_id 
    # 创建模型并以半精度形式加载
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code = True, 
        torch_dtype = torch.half, 
        device_map = "auto"
    )
    # model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法

    # 将数据集变化为 token 形式
    tokenized_id = ds.map(process_func, remove_columns = ds.column_names)

    # 加载 Lora 参数
    peft_model = get_peft_model(model, config)

    # 使用 trainer 训练
    trainer = Trainer(
        model = peft_model,
        args = args,
        train_dataset = tokenized_id,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, padding = True),
    )
    trainer.train()

    # 模型推理
    response, history = peft_model.chat(
        tokenizer, 
        "你是谁", 
        history = [], 
        system = "现在你要扮演皇帝身边的女人--甄嬛."
    )
    print(response)

if __name__ == "__main__":
    main()
