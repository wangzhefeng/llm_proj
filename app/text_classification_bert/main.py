# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-09
# * Version     : 0.1.110900
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

import numpy as np
import pandas as pd
import torch
from torch import nn 
from torch.utils.data import DataLoader
import datasets
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification 
import evaluate
from transformers import pipeline

from torchkeras import KerasModel

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data
# ------------------------------
df = pd.read_csv("./dataset/waimai_10k.csv")
# print(df.head())
ds = datasets.Dataset.from_pandas(df)
ds = ds.shuffle(42)  # 打乱顺序
ds = ds.rename_columns({
    "review": "text",
    "label": "labels",
})
print(ds[0:6])

# ------------------------------
# 文本分词
# tokenizer 可以使用 __call__, encode, encode_plus, batch_encode_plus 等方法编码；使用 decode, batch_decode 等方法进行解码
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
print(tokenizer)

"""
# 编码测试
text_codes = tokenizer(
    text = "晚了半小时，七元套餐饮料就给的罐装的可乐，真是可以",
    text_pair = None,
    max_length = 100,  # 为空则默认为模型最大长度，BERT: 512, GPT: 1024
    truncation = True,
    padding = "do_not_pad",  # 可选: "longest"/"max_length"/"do_not_pad"
)
print(text_codes)

# 解码测试
print(tokenizer.decode(text_codes["input_ids"][0]))
print()
print(tokenizer.decode(text_codes["input_ids"][1]))
print()
print(tokenizer.batch_decode(text_codes["input_ids"]))

# token & ids
tokens = tokenizer.tokenize(ds["text"][0])
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
"""

# 分词
ds_encoder = ds.map(
    lambda example: tokenizer(
        text = example["text"], 
        max_length=50, 
        truncation=True, 
        padding="max_length"
    ),
    batched=True,
    batch_size=20,
    num_proc=1  # 支持批处理和多进程 map
)

# ------------------------------
# 构建 DataLoader
# ------------------------------
# 转换成 PyTorch 中的 tensor
ds_encoder.set_format(
    type = "torch", 
    columns=[
        "input_ids", "attention_mask", "token_type_ids", "labels"
    ],
)
# ds_encoder.reset_format()
print(ds_encoder[0])

# 分割成训练集和测试集
ds_train_val, ds_test = ds_encoder.train_test_split(test_size=0.2).values()
ds_train, ds_val = ds_train_val.train_test_split(test_size=0.2).values()

# 在 collate_fn 中可以做动态批处理(dynamic batching)
def collate_fn(examples):
    return tokenizer.pad(examples)  # return_tensors="pt"
# or
collate_fn = DataCollatorWithPadding(tokenizer = tokenizer)

dl_train = DataLoader(ds_train, batch_size = 16, collate_fn = collate_fn)
dl_val = DataLoader(ds_val, batch_size = 16,  collate_fn = collate_fn)
dl_test = DataLoader(ds_test, batch_size = 16,  collate_fn = collate_fn)

for batch in dl_train:
    break

# ------------------------------
# 定义模型
# ------------------------------

#加载模型 (会添加针对特定任务类型的Head)
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-chinese', 
    num_labels = 2
)
print(dict(model.named_children()).keys())

# ------------------------------
# 训练模型
# ------------------------------
# 需要修改 StepRunner 以适应 transformers 的数据集格式
class StepRunner:
    
    def __init__(self, net, loss_fn, accelerator, stage = "train", metrics_dict = None, 
                 optimizer = None, lr_scheduler = None
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        self.accelerator = accelerator
        if self.stage=='train':
            self.net.train() 
        else:
            self.net.eval()
    
    def __call__(self, batch):
        out = self.net(**batch)
        #loss
        loss= out.loss
        #preds
        preds =(out.logits).argmax(axis=1) 
        #backward()
        if self.optimizer is not None and self.stage=="train":
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
        all_loss = self.accelerator.gather(loss).sum()
        labels = batch['labels']
        acc = (preds==labels).sum()/((labels>-1).sum())
        all_acc = self.accelerator.gather(acc).mean()
        # losses
        step_losses = {
            self.stage + "_loss": all_loss.item(), 
            self.stage + '_acc': all_acc.item()
        }
        # metrics
        step_metrics = {}
        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses,step_metrics

KerasModel.StepRunner = StepRunner

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-5)

# model
keras_model = KerasModel(
    model,
    loss_fn = None,
    optimizer = optimizer
)

keras_model.fit(
    train_data = dl_train,
    val_data= dl_val,
    ckpt_path='bert_waimai.pt',
    epochs=100,
    patience=10,
    monitor="val_acc", 
    mode="max",
    plot = True,
    wandb = False,
    quiet = True
)

# ------------------------------
# 模型评估
# ------------------------------
metric = evaluate.load("accuracy")

model.eval()
dl_test = keras_model.accelerator.prepare(dl_test)
for batch in dl_test:
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

# ------------------------------
# 模型使用
# ------------------------------
texts = [
    "味道还不错，下次再来", 
    "这他妈也太难吃了吧", 
    "感觉不是很新鲜",
    "还行我家狗狗很爱吃"
]
batch = tokenizer(texts, padding = True, return_tensors = "pt")
batch = {
    k: v.to(keras_model.accelerator.device) 
    for k,v in batch.items()
}
logits = model(**batch).logits 
scores = nn.Softmax(dim = -1)(logits)[:,-1]
print(scores)


# 也可以用 pipeline 把 tokenizer 和 model 组装在一起
classifier = pipeline(
    task = "text-classification",
    tokenizer = tokenizer,
    model = model.cpu()
)
classifier("挺好吃的哦")


# ------------------------------
# 模型保存
# ------------------------------
# model save
model.config.id2label = {
    0: "差评",
    1: "好评"
}
model.save_pretrained("waimai_10k_bert")
tokenizer.save_pretrained("waimai_10k_bert")

# model load
classifier = pipeline(task = "text-classification", model="waimai_10k_bert")

# model use
classifier([
    "味道还不错，下次再来",
    "我去，吃了我吐了三天"
])




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
