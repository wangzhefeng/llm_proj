# -*- coding: utf-8 -*-

# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-27
# * Version     : 1.0.092700
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

import torch

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# I/O 配置，用于定义输出目录和训练时的日志记录与评估设置
out_dir = "./tiny-universe/TinyLlama/output"  # 模型输出保存路径
eval_interval = 2000  # 评估间隔步数
log_interval = 1  # 日志记录间隔步数
eval_iters = 100  # 每次评估时迭代的步数
eval_only = False  # 如果为 True，脚本在第一次评估后立即退出
always_save_checkpoint = False  # 如果为True，在每次评估后总是保存检查点
init_from = "scratch"  # 可以选择从头开始训练（'scratch'）或从已有的检查点恢复（'resume'）

# 数据配置
batch_size = 128  # 每个微批次的样本数量，如果使用梯度累积，实际批次大小将更大
max_seq_len = 256  # 最大序列长度
vocab_size = 4096  # 自定义词汇表大小

# 模型配置
dim = 288  # 模型的隐藏层维度
n_layers = 8  # Transformer 的层数
n_heads = 8  # 注意力头的数量
n_group = 4  # 模型分组
multiple_of = 32  # 在某些层的维度必须是该数的倍数
dropout = 0.0  # Dropout 概率

# AdamW 优化器配置
gradient_accumulation_steps = 4  # 梯度累积步数，用于模拟更大的批次
learning_rate = 5e-4  # 最大学习率
max_iters = 100000  # 总的训练迭代次数
weight_decay = 1e-1  # 权重衰减系数
beta1 = 0.9  # AdamW 优化器的 β1 参数
beta2 = 0.95  # AdamW 优化器的 β2 参数
grad_clip = 1.0  # 梯度裁剪阈值，0表示不裁剪
# 学习率衰减配置
decay_lr = True  # 是否启用学习率衰减
warmup_iters = 1000  # 学习率预热的步数


# 系统设置
device = "cuda:0"  # 设备选择：'cpu'，'cuda'，'cuda:0' 等
dtype = "bfloat16"  # 数据类型：'float32'，'bfloat16'，'float16'

# ------------------------------
# 配置参数
# ------------------------------
# 获取配置参数的键值对，便于后续的日志记录
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
# 保存配置到字典中，便于日志记录
config = {k: globals()[k] for k in config_keys}

# ------------------------------
# 固定一些超参数的默认值
# ------------------------------
lr_decay_iters = max_iters  # 学习率衰减步数，设置为等于最大迭代步数
min_lr = 0.0  # 最小学习率，建议为学习率的十分之一
vocab_source = 'custom'  # 词汇表来源
master_process = True  # 用于区分主进程
seed_offset = 0  # 随机种子偏移量
ddp_world_size = 1  # 分布式数据并行的世界大小
tokens_per_iter = batch_size * max_seq_len  # 每次迭代处理的 token 数

# 设置随机种子，确保可重复性
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # 允许在matmul上使用tf32
torch.backends.cudnn.allow_tf32 = True  # 允许在cudnn上使用tf32
device_type = "cuda" if "cuda" in device else "cpu"  # 用于自动选择设备类型
ptdtype = torch.float16  # 设置训练时使用的数据类型


print(config_keys)
print()
print(config)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
