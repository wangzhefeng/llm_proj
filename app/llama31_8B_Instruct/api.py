# -*- coding: utf-8 -*-

# ***************************************************
# * File        : api.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-16
# * Version     : 0.1.081622
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
import datetime

import uvicorn
from fastapi import FastAPI, Request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

# 设置设备参数
DEVICE = "cuda"  # 使用 CUDA
DEVICE_ID = "0"  # CUDA 设备 ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合 CUDA 设备信息

# 清理 GPU 内存函数
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()  # 清空 CUDA 缓存
            torch.cuda.ipc_collect()  # 收集 CUDA 内存碎片

# 创建 FastAPI 应用
app = FastAPI()

# 处理 POST 请求的端点
@app.post("/")
async def create_item(request: Request):
    # 声明全局变量以便在函数内部使用模型和分词器
    # global model, tokenizer
    # 加载预训练的分词器和模型
    model_name_or_path = "D:\projects\llms_proj\llm_proj\downloaded_models\LLM-Research\Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        use_fast = False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        device_map = "auto", 
        torch_dtype = torch.bfloat16
    )

    # 获取 POST 请求的 JSON 数据
    json_post_raw = await request.json()
    # 将 JSON 数据转换为字符串
    json_post = json.dumps(json_post_raw)
    # 将字符串转换为 Python 对象
    json_post_list = json.loads(json_post)
    # 获取请求中的提示
    prompt = json_post_list.get("prompt")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    # 调用模型进行对话生成
    input_ids = tokenizer.apply_chat_template(
        messages, 
        tokenize = False, 
        add_generation_prompt = True
    )
    model_inputs = tokenizer(
        [input_ids],
        return_tensors = "pt",
    ).to(DEVICE)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens = 512)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)[0]

    # 构建响应 JSON
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time,
    }

    # 构建日志信息
    log = f"[{time}], prompt:{prompt}, response:{repr(response)}"
    print(log)

    # 执行 GPU 内存清理
    torch_gc()

    return answer




# 测试代码 main 函数
def main(): 
    # 启动 FastAPI 应用，在指定端口和主机上启动应用
    # 用 6006 端口可以将 autodl 的端口映射到本地，从而在本地使用 API
    uvicorn.run(app, host = "0.0.0.0", port = 6006, workers = 1)

if __name__ == "__main__":
    main()
