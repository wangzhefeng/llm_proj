# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-09
# * Version     : 0.1.110916
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
from typing import List, Dict
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import datasets
from transformers import (
    BertTokenizer, 
    DataCollatorForTokenClassification,
    BertForTokenClassification,
)
from torchmetrics import Accuracy
from transformers import pipeline

from torchkeras import KerasModel

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# data
# ------------------------------
def get_text_data(data_dir: str):
    """
    train and validation data
    """ 
    df_train = pd.read_json(os.path.join(data_dir, "train.json"), lines = True)
    df_test = pd.read_json(os.path.join(data_dir, "test.json"), lines = True)
    df_val = pd.read_json(os.path.join(data_dir, "dev.json"), lines = True)

    return df_train, df_test, df_val


def get_label_id():
    # entity
    entities = [
        "address",
        "book",
        "company",
        "game",
        "government",
        "movie",
        "name",
        "organization",
        "position",
        "scene",
    ]
    print(f"entity number: {len(entities)}\n")
    # labels
    label_names = ["O"] + ["B-" + x for x in entities] + ["I-" + x for x in entities]
    # print(f"label_names: {label_names}", "\n")
    # id to label & label to id
    id_to_label = {i: label for i, label in enumerate(label_names)}
    label_to_id = {label: i for i, label in enumerate(label_names)}
    
    return id_to_label, label_to_id


# ------------------------------
# 文本分词
# ------------------------------
def get_tokenizer(model_name: Dict): 
    """
    分词
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)

    return tokenizer


# ------------------------------
# 对齐标签
# ------------------------------
def get_char_label(text, label) -> List[str]:
    """
    把 label 格式转化成字符级别的 char_label

    Args:
        text (_type_): _description_
        label (_type_): _description_

    Returns:
        _type_: _description_
    """
    char_label = ["O" for x in text]
    for tp, dic in label.items():
        for word, idxs in dic.items():
            idx_start, idx_end = idxs[0][0], idxs[0][1]
            char_label[idx_start] = "B-" + tp
            char_label[(idx_start + 1):(idx_end+1)] = [
                "I-" + tp 
                for x in range(idx_start + 1, idx_end + 1)
            ]
    
    return char_label


def get_token_label(text, char_label, tokenizer):
    """
    将 char_label 对齐到 token_label

    Args:
        text (_type_): _description_
        char_label (_type_): _description_
        tokenizer (_type_): _description_
    """
    # 分词
    tokenized_input = tokenizer(text)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
    # text/tokens/char_label iter
    iter_text = iter(text.lower())
    iter_tokens = iter(tokens)
    iter_char_label = iter(char_label)
    char = next(iter_text)
    token = next(iter_tokens)
    char_tp = next(iter_char_label)
    # token labels
    token_labels = []
    while True:
        # 单个字符 token（如汉字）直接赋给对应字符 token
        if len(token) == 1:
            assert token == char
            token_labels.append(char_tp)
            # update char and char_tp
            try:
                char = next(iter_text)
                char_tp = next(iter_char_label)
            except StopIteration:
                pass
        # 特殊字符
        elif token in tokenizer.special_tokens_map.values() and token != "[UNK]":
            token_labels.append("O")
        # token 为 "[UNK]"
        elif token == "[UNK]":
            token_labels.append(char_tp)
            # 重新对齐
            try:
                token = next(iter_tokens)
            except:
                break
            # update char and char_tp
            if token not in tokenizer.special_tokens_map.values():
                while char != token[0]:
                    try:
                        char = next(iter_text)
                        char_tp = next(iter_char_label)
                    except StopIteration:
                        pass
            continue
        # 其他长度大于 1 的 token，如英文 token
        else:
            token_label = char_tp
            # 移除因 subword 引入的 "##" 符号
            token = token.replace("##", "")
            for c in token:
                assert c == char or char not in tokenizer.vocab
                if token_label != "O":
                    token_label = char_tp
                # update char and char_tp
                try:
                    char = next(iter_text)
                    char_tp = next(iter_char_label)
                except StopIteration:
                    pass
            token_labels.append(token_label)
        
        # 停止条件
        try:
            token = next(iter_tokens)
        except StopIteration:
            break
    
    assert len(token_labels) == len(tokens)
    return token_labels


# ------------------------------
# 构建管道
# ------------------------------
def make_sample(tokenizer, text, label, label_to_id):
    # 分词
    sample = tokenizer(text)
    # char label
    char_label = get_char_label(text, label)
    # token label
    token_labels = get_token_label(text, char_label, tokenizer)
    # labels
    sample["labels"] = [label_to_id[x] for x in token_labels]

    return sample


def make_dataloader(tokenizer, label_to_id, df_train, df_val, batch_size: int = 8):
    train_samples = [
        make_sample(tokenizer, text, label, label_to_id) 
        for text, label in tqdm(list(zip(df_train["text"], df_train["label"])))
    ]
    val_samples = [
        make_sample(tokenizer, text, label, label_to_id) 
        for text, label in tqdm(list(zip(df_val["text"], df_val["label"])))
    ]

    ds_train = datasets.Dataset.from_list(train_samples)
    ds_val = datasets.Dataset.from_list(val_samples)

    data_collator = DataCollatorForTokenClassification(tokenizer = tokenizer)
    dl_train = DataLoader(ds_train, batch_size = batch_size, collate_fn = data_collator)
    dl_val = DataLoader(ds_val, batch_size = batch_size, collate_fn = data_collator)

    return dl_train, dl_val


# ------------------------------
# 模型构建
# ------------------------------
def model_build(model_name: str, id2label: Dict, label2id: Dict):
    # Bert token 分类模型
    net = BertForTokenClassification.from_pretrained(
        model_name,
        id2label = id2label,
        label2id = label2id,
    )
    # 冻结 Bert 基模型参数
    for param in net.bert.parameters():
        param.requires_grad_(False) 

    return net


# ------------------------------
# 模型训练
# ------------------------------
class StepRunner:

    def __init__(self, 
                 net, 
                 loss_fn, 
                 accelerator, 
                 stage = "train", 
                 metrics_dict = None, 
                 optimizer = None, 
                 lr_scheduler = None):
        self.net = net
        self.loss_fn = loss_fn
        self.accelerator = accelerator
        self.stage = stage
        self.metrics_dict = metrics_dict
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        # model train or eval
        if self.stage == "train":
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch):
        # model
        out = self.net(**batch)
        # true labels
        labels = batch['labels']
        # preds
        preds = (out.logits).argmax(axis = 2)
        # loss
        loss = out.loss
        # backward
        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
        # losses
        all_loss = self.accelerator.gather(loss).sum()
        # precision & recall
        precision = (((preds > 0) & (preds == labels)).sum()) / (torch.maximum((preds > 0).sum(), torch.tensor(1.0).to(preds.device)))
        recall = (((labels > 0) & (preds == labels)).sum()) / (torch.maximum((labels > 0).sum(), torch.tensor(1.0).to(labels.device)))
        all_precision = self.accelerator.gather(precision).mean()
        all_recall = self.accelerator.gather(recall).mean()
        f1 = 2 * all_precision * all_recall / torch.maximum(all_recall + all_precision, torch.tensor(1.0).to(labels.device))
        
        # losses
        step_losses = {
            f"{self.stage}_loss": all_loss.item(),
            f"{self.stage}_precision": all_precision.item(),
            f"{self.stage}_recall": all_recall.item(),
            f"{self.stage}_f1": f1.item()
        }
        # metrics
        step_metrics = {}
        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        
        return step_losses, step_metrics


def model_training(net, dl_train, dl_val, epochs: int, ckpt_dir: str):
    # step runner
    KerasModel.StepRunner = StepRunner 
    # optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr = 3e-5)
    # keras model
    keras_model = KerasModel(
        net, 
        loss_fn = None,
        optimizer = optimizer,
    )
    keras_model.fit(
        train_data = dl_train,
        val_data = dl_val,
        epochs = epochs,
        ckpt_path = ckpt_dir,
        patience = 5,
        monitor = "val_f1",
        mode = "max",
        plot = True,
        wandb = False,
    )
    
    return keras_model


# ------------------------------
# 模型评估
# ------------------------------
def model_validating(keras_model, dl_val, net):
    acc = keras_model.accelerator.prepare(Accuracy(task = "multiclass", num_classes = 21))
    dl_test = keras_model.accelerator.prepare(dl_val)
    net = keras_model.accelerator.prepare(net)

    for batch in tqdm(dl_test):
        with torch.no_grad():
            outputs = net(**batch)
        
        labels = batch["labels"]
        labels[labels < 0] = 0
        # prediction
        preds = (outputs.logits).argmax(axis = 2)
        acc.update(preds, labels)
    # 这里的 acc 包括了 "O" 的分类结果，存在高估
    acc.compute()
    
    return net


# ------------------------------
# 模型使用
# ------------------------------
def model_use(net, tokenizer, text: str):
    recognizer = pipeline(
        "token-classification",
        model = net,
        tokenizer = tokenizer,
        aggregation_strategy = "simple",
        device = "cpu",
    )
    net.to("cpu")
    result = recognizer(text)

    return result


# ------------------------------
# 模型保存
# ------------------------------
def model_save(net, tokenizer, model_dir: str):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # model save
    net.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def model_load(model_dir: str, text: str = None):
    # model load
    recognizer = pipeline(
        "token-classification",
        model = model_dir,
        aggregation_strategy = "simple",
        device = "cpu"
    )
    result = recognizer(text)

    return result




# 测试代码 main 函数
def main():
    # ------------------------------
    # parameters
    # ------------------------------
    # data dir
    data_dir = "./dataset/cluener_public/" 
    # model name
    model_name = "bert-base-chinese"
    # batch size
    batch_size = 8
    # epochs
    epochs = 5
    # checkpoints dir
    ckpt_dir = f"./app/ner_bert/checkpoints/ner-bert-mn{model_name}-bs{batch_size}-ep{epochs}.pt"
    # model save dir
    model_dir = f"./app/ner_bert/pt_save_pretrained/ner-bert-mn{model_name}-bs{batch_size}-ep{epochs}/"
    # ------------------------------
    # data
    # ------------------------------
    print("-" * 80)
    print("Run get_text_data...")
    print("-" * 80)
    df_train, df_test, df_val = get_text_data(data_dir = data_dir)
    # df train
    print(f"df_train head:\n{df_train.head()}")
    # text
    # text = df_train["text"][43]
    # print(f"text: {text}")
    # print(f"text length: {len(text)}")
    # label
    # label = df_train["label"][43]
    # print(f"label: {label}", "\n")

    # label name & id
    print("-" * 80)
    print("Run get_label_id()...")
    print("-" * 80)
    id_to_label, label_to_id = get_label_id()
    print(f"id_to_label: {id_to_label}\n")
    print(f"label_to_id: {label_to_id}")
    # ------------------------------
    # 分词
    # ------------------------------
    print("-" * 80)
    print("Run get_tokenizer()...")
    print("-" * 80)
    tokenizer = get_tokenizer(model_name = model_name)
    # tokenizer
    print(f"tokenizer:\n{tokenizer}")
    # 分词
    # tokenized_input = tokenizer(text)
    # print(f"tokenized_input: {tokenized_input}")
    # print(f"tokenized_input input_ids: {tokenized_input['input_ids']}")
    # print(f"tokenized_input length: {len(tokenized_input['input_ids'])}", "\n")
    # 分词逆转换
    # tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
    # print(f"tokens: {tokens}")
    # print(f"tokens length: {len(tokens)}", "\n")
    # ------------------------------
    # 标签对齐
    # ------------------------------
    # char label
    # char_label = get_char_label(text = text, label = label)
    # print(f"char_label: {char_label}")
    # print(f"char_label length: {len(char_label)}", "\n")
    # for char, char_tp in zip(text, char_label):
    #     print(char + "\t" + char_tp)
    
    # token labels
    # token_labels = get_token_label(text = text, char_label = char_label, tokenizer = tokenizer)
    # print(f"token_labels: {token_labels}")
    # print(f"token_labels length: {len(token_labels)}")
    # for t, t_label in zip(tokens, token_labels):
    #     print(t, "\t", t_label)
    # ------------------------------
    # 构建数据管道
    # ------------------------------
    print("-" * 80)
    print("Run make_dataloader()...")
    print("-" * 80)
    dl_train, dl_val = make_dataloader(
        tokenizer = tokenizer, 
        label_to_id = label_to_id, 
        df_train = df_train,
        df_val = df_val,
        batch_size = batch_size,
    )
    # ------------------------------
    # 模型构建
    # ------------------------------
    print("-" * 80)
    print("Run model_build()...")
    print("-" * 80)
    net = model_build(model_name = model_name, id2label = id_to_label, label2id = label_to_id)
    # model label
    # print(f"net.config.num_labels: {net.config.num_labels}")
    # 模型计算
    # for batch in dl_train:
    #     break
    # out = net(**batch)
    # print(f"losses: {out.loss}")
    # print(f"logits shape: {out.logits.shape}")
    # ------------------------------
    # 模型训练
    # ------------------------------
    print("-" * 80)
    print("Run model_training()...")
    print("-" * 80)
    keras_model = model_training(net = net, dl_train = dl_train, dl_val = dl_val, epochs = epochs, ckpt_dir = ckpt_dir)
    print("-" * 80)
    print("Run model_validating()...")
    print("-" * 80)
    net = model_validating(keras_model = keras_model, dl_val = dl_val, net = net)
    print("-" * 80)
    print("Run model_use()...")
    print("-" * 80)
    result = model_use(net = net, tokenizer = tokenizer, text = "小明对小红说，“你听说过安利吗？")
    print(result)
    print("-" * 80)
    print("Run model_save()...")
    print("-" * 80)
    model_save(net = net, tokenizer = tokenizer, model_dir = model_dir)
    print("-" * 80)
    print("Run model_load()...")
    print("-" * 80)
    result = model_load(model_dir =  model_dir, text = "小明对小红说，“你听说过安利吗？")
    print(result)

if __name__ == "__main__":
    main()
