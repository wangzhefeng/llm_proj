# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LLM_Lora_tuning_SwanLab.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-11
# * Version     : 0.1.111113
# * Description : description
# * Link        : Data: https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT
# *                     https://github.com/FudanDISC/DISC-LawLLM
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
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from swanlab.integration.transformers import SwanLabCallback
import swanlab

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}.")


# ------------------------------
# 微调数据加载和预处理
# ------------------------------
def __process_func(example, tokenizer):
    """
    微调数据格式化

    Qwen2 采用的 Prompt Template 格式如下：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    你是谁？<|im_end|>
    <|im_start|>assistant
    我是一个有用的助手。<|im_end|>
    """
    # 分词器会将一个中文字切分为多个 token，因此需要放开一些最大长度，保证数据的完整性
    MAX_LENGTH = 384
    # 指令集构建
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens = False,  # 不在开头加 special_tokens
    )
    response = tokenizer(
        f"{example['output']}", 
        add_special_tokens = False
    )
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


# ------------------------------
# 模型加载和微调训练
# ------------------------------
def get_tokenizer_model(model_path):
    """
    加载 tokenizer 和半精度模型
    """
    # 加载本地 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast = False,
        trust_remote_code = True
    )
    # tokenizer.pad_token = tokenizer.eos_token
    print(f"tokenizer:\n{tokenizer}")

    # 加载本地 LLM 模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,  
        torch_dtype = torch.bfloat16, 
        device_map = "auto",
        trust_remote_code = True,
    )
    # 开启梯度检查点时，执行以下方法
    model.enable_input_require_grads()
    print(f"model:\n{model}")
    print(f"model type:{model.dtype}")
    
    return tokenizer, model


def model_predicting(messages, model, tokenizer, lora_path, lora_config):
    """
    模型推理
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


class LawSwanLabCallback(SwanLabCallback):

    def __init__(self, test_df, peft_model, tokenizer, lora_path, lora_config):
        super(LawSwanLabCallback, self).__init__()
        # params
        self.test_df = test_df
        self.peft_model = peft_model
        self.tokenizer = tokenizer
        self.lora_path = lora_path
        self.lora_config = lora_config

    def on_train_begin(self, args, state, control, model = None, **kwargs):
        """
        训练开始阶段，取三条主观评测
        """
        if not self._initialized:
            self.setup(args, state, model, **kwargs)
        print("训练开始...\n未开始微调，先取 3 条主观评测：")
        test_text_list = []
        for index, row in self.test_df[:3].iterrows():
            # messages(prompt template)
            instruction = row["instruction"]
            input_value = row["input"]
            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"},
            ]
            response = model_predicting(messages, self.peft_model, self.tokenizer, self.lora_path, self.lora_config)
            messages.append(
                {"role": "assistant", "content": f"{response}"}
            )
            # result text
            result_text = f"""
                【Q】{messages[1]['content']}\n
                【LLM】{messages[2]['content']}\n
            """
            test_text_list.append(swanlab.Text(result_text, caption = response))
        # log
        swanlab.log({"Prediction": test_text_list}, step = 0)

    def on_train_end(self, args, state, control, model = None, tokenizer = None, **kwargs):
        """
        测试阶段
        """
        print("测试阶段...")
        test_text_list = []
        for index, row in self.test_df.iterrows():
            # messages(prompt template)
            instruction = row["instruction"]
            input_value = row["input"]
            ground_truth = row["output"]
            messages = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_value}"},
            ]
            response = model_predicting(messages, self.peft_model, self.tokenizer, self.lora_path, self.lora_config)
            messages.append(
                {"role": "assistant", "content": f"{response}"}
            )
            # result text
            if index == 0:
                print("epoch", round(state.epoch), "主观评测：")
            result_text = f"""
                【Q】{messages[1]['content']}\n
                【LLM】{messages[2]['content']}\n
                【GT】 {ground_truth}
            """
            test_text_list.append(swanlab.Text(result_text, caption = response))
        # log
        swanlab.log({"Prediction": test_text_list}, step = round(state.epoch))


def model_training(model, tokenizer, train_dataset, test_df, lora_path):
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
        r = 64,  # Lora 秩
        lora_alpha = 16,  # Lora alpha
        lora_dropout = 0.1,  # dropout 比例
    )
    print(lora_config)

    # 创建 Peft 模型
    peft_model = get_peft_model(model, lora_config)
    print(peft_model.print_trainable_parameters())

    # 配置 LoRA 训练参数
    args = TrainingArguments(
        output_dir = lora_path,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        logging_steps = 10,
        num_train_epochs = 2,
        save_steps = 100,  # 快速演示设置 10，建议设置为 100
        learning_rate = 1e-4,
        save_on_each_node = True,
        gradient_checkpointing = True,
        report_to = "none",
    )

    # swanlab callback
    swanlab_callback = LawSwanLabCallback(
        project = "Qwen2.5-LoRA-Law",
        experiment_name = "7b",
        config = {
            "model": "https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct",
            "dataset": "https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT",
            "github": "https://github.com/datawhalechina/self-llm",
            "system_prompt": "你是一个法律专家，请根据用户的问题给出专业的回答",
            "lora_rank": 64,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
        },
        test_df = test_df, 
        peft_model = peft_model, 
        tokenizer = tokenizer, 
        lora_path = lora_path, 
        lora_config = lora_config,
    )
    
    # 使用 Trainer 训练
    trainer = Trainer(
        model = peft_model,
        args = args,
        train_dataset = train_dataset,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, padding = True),
        callbacks = [swanlab_callback],
    )
    trainer.train()
    
    # 在 Jupyter Notebook 中运行时要停止 SwanLab 记录，需要调用 swanlab.finish()
    swanlab.finish()




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

if __name__ == "__main__":
    main()
