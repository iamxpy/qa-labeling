# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import

import argparse
import logging
import os
import random
from io import open
import pandas as pd
import numpy as np
import torch
import gc
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
from scipy.stats import spearmanr
from modeling_bert import BertForSequenceClassification_linear, BertConfig
from bert_utils import AdamW, WarmupLinearSchedule
from bert_utils import BertTokenizer
from stratifiers import MultilabelStratifiedKFold
from math import ceil, floor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification_linear, BertTokenizer),
}
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, q_title, q_body, answer, label=None):
        self.q_title = q_title
        self.q_body = q_body
        self.answer = answer
        self.label = label


class InputFeatures(object):
    def __init__(self, choices_features, label):
        self.choices_features = choices_features
        self.label = label


def read_examples(df: pd.DataFrame, feature_labels: list, label_columns: list = None):
    columns = list(feature_labels)
    if label_columns is not None:
        columns += label_columns
    examples = []
    for val in df[columns].values:
        if label_columns is not None:
            examples.append(InputExample(q_title=val[0], q_body=val[1], answer=val[2], label=val[3:]))
        else:
            examples.append(InputExample(q_title=val[0], q_body=val[1], answer=val[2]))
    return examples


def _tokenize_trim(title, question, answer, max_sequence_length, tokenizer,
                   t_max_len=30, q_max_len=239, a_max_len=239):
    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)

    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len + q_len + a_len + 4) > max_sequence_length:

        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len) / 2)
            q_max_len = q_max_len + ceil((t_max_len - t_len) / 2)
        else:
            t_new_len = t_max_len

        if a_max_len > a_len:
            a_new_len = a_len
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len

        if t_new_len + a_new_len + q_new_len + 4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d"
                             % (max_sequence_length, (t_new_len + a_new_len + q_new_len + 4)))

        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]

    return t, q, a


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))  # [PAD]对应segment id为0


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))  # [PAD]对应vocab id为0
    return input_ids


def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    print('Preprocessing ......')
    input_ids, input_masks, input_segments = [], [], []
    for _, row in df[columns].iterrows():
        t, q, a = row.question_title, row.question_body, row.answer

        t, q, a = _tokenize_trim(t, q, a, max_sequence_length, tokenizer)

        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32),
            np.asarray(input_segments, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _predict(args, model, X, Y):
    # Validation data
    eval_ids, eval_masks, eval_segments = X
    eval_labels = Y
    # 每个epoch的结尾进行一次Evaluation
    predictions = []
    gold_labels = []
    eval_ids_tensor = torch.tensor(eval_ids, dtype=torch.long)
    eval_masks_tensor = torch.tensor(eval_masks, dtype=torch.long)
    eval_segments_tensor = torch.tensor(eval_segments, dtype=torch.long)
    eval_labels_tensor = torch.tensor(eval_labels, dtype=torch.float32)

    eval_data = TensorDataset(eval_ids_tensor, eval_masks_tensor, eval_segments_tensor,
                              eval_labels_tensor)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss, eval_rho, nb_eval_steps, nb_eval_examples = 0, 0, 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)
        label_ids = label_ids.to(args.device)

        with torch.no_grad():
            batch_eval_loss, batch_logits = model(input_ids=input_ids, token_type_ids=segment_ids,
                                                  attention_mask=input_mask, labels=label_ids)
        batch_preds = torch.sigmoid(batch_logits).cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        # 将所有batch得到的预测标签拼接起来, shape: [eval_batch_size,num_class]
        predictions.append(batch_preds)
        # 将所有batch的真实标签拼接起来, shape: [eval_batch_size,num_class]
        gold_labels.append(label_ids)
        eval_loss += batch_eval_loss.mean().item()
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    gold_labels = np.concatenate(gold_labels, 0)
    predictions = np.concatenate(predictions, 0)
    eval_loss = eval_loss / nb_eval_steps
    eval_rho = compute_spearmanr(gold_labels, predictions)
    return predictions, eval_loss, eval_rho


def train_and_predict(model, train_data, valid_data, args):
    # Training data
    train_ids, train_masks, train_segments = train_data[0]
    train_labels = train_data[1]

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # 一般只对weight做正则化而不管bias，至于LayerNorm层的参数为啥不用正则化，还不知道
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                     t_total=args.epochs * ceil(train_ids.shape[0] / args.train_batch_size))

    tr_loss = 0  # 一个epoch的累积loss，用于计算平均loss
    nb_tr_steps = 0  # 一个epoch内已经前向计算的batch数
    val_rhos = []  # 记录validation的rho，用于后期voting

    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}/{args.epochs}")
        # Prepare data loader
        train_ids_tensor = torch.tensor(train_ids, dtype=torch.long)
        train_mask_tensor = torch.tensor(train_masks, dtype=torch.long)
        train_segments_tensor = torch.tensor(train_segments, dtype=torch.long)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)

        train_data = TensorDataset(train_ids_tensor, train_mask_tensor, train_segments_tensor, train_labels_tensor)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        bar = tqdm(total=train_ids.shape[0], ncols=100)
        model.train()  # Sets the module in training mode
        for input_ids, input_mask, segment_ids, label_ids in train_dataloader:
            actual_batch_size = input_ids.shape[0]  # 用于更新tqdm, 最后一个batch大小可能不等于batch_size
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            segment_ids = segment_ids.to(args.device)
            label_ids = label_ids.to(args.device)
            loss, _ = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                            labels=label_ids)  # 训练集上得到的logits直接丢弃，不需要计算指标
            del input_ids, input_mask, segment_ids, label_ids
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            tr_loss += loss.item()  # todo 用tensorboard记录这些指标
            # tr_loss是真实的batch累积loss，除以training的batch数得到平均batch loss
            train_loss = round(tr_loss / (nb_tr_steps + 1), 4)

            bar.set_description(f"loss {round(train_loss, 4)}")
            nb_tr_steps += 1  # 记录一个epoch内已经前向传播的batch个数
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()  # 清空梯度
            bar.update(actual_batch_size)

        _, eval_loss, eval_rho = _predict(args, model, valid_data[0], valid_data[1])
        val_rhos.append(eval_rho)
        print(f"\n\nvalidation loss: {round(eval_loss, 4)}, rho:{round(float(eval_rho), 4)}\n")
        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, f"fold-{args.fold}-epoch-{epoch}.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

        bar.close()
        # epoch结束时重置这两个参数
        tr_loss = 0
        nb_tr_steps = 0

    return val_rhos


def get_model(args, config):
    # Prepare model
    # todo 修改网络架构之后需要修改model类
    # todo 加的配置项如果想要传给model类，需要修改此处的from_pretrained方法
    model = BertForSequenceClassification_linear.from_pretrained(args.model_name_or_path, args, config=config)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model


def get_tokenizer_and_config(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
    return tokenizer, config


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.model_type = "bert"
    args.model_name_or_path = "../input/bert-base-uncased"
    args.do_lower_case = True
    args.data_dir = "../input/google-quest-challenge/"
    args.output_dir = "../output/"
    args.num_fold = 10
    args.epochs = 7
    args.eval_steps = 3
    args.max_seq_length = 512
    args.split_num = 1
    args.lstm_hidden_size = 512
    args.lstm_layers = 1
    args.lstm_dropout = 0.1
    args.dropout = 0.2
    args.per_gpu_train_batch_size = 8
    args.warmup_steps = 0
    args.per_gpu_eval_batch_size = 64
    args.learning_rate = 3e-5
    args.warmup_strategy = None
    args.adam_epsilon = 1e-7
    args.weight_decay = 0
    args.freeze = 0
    args.max_grad_norm = 1.0
    args.seed = 2019
    args.no_cuda = False
    args.do_train = True
    # Setup GPU
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    # Set seed
    set_seed(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df_train = pd.read_csv(args.data_dir + 'train.csv')
    print('train shape =', df_train.shape)
    # Read data
    # if args.task == "question":
    #     input_categories = list(df_train.columns[[1, 2]])
    #     output_categories = list(df_train.columns[11:32])  # 21 categories
    #     num_labels = 21
    # elif args.task == "answer":
    #     input_categories = list(df_train.columns[[1, 5]])
    #     output_categories = list(df_train.columns[32:])  # 9 categories
    #     num_labels = 9
    # else:
    #     raise ValueError("Task must be one of 'question' and 'answer'")
    input_categories = list(df_train.columns[[1, 2, 5]])
    output_categories = list(df_train.columns[11:])  # 30 categories
    args.num_labels = 30

    tokenizer, config = get_tokenizer_and_config(args)

    inputs = compute_input_arays(df_train, input_categories, tokenizer, args.max_seq_length)
    outputs = compute_output_arrays(df_train, output_categories)

    kf = MultilabelStratifiedKFold(n_splits=args.num_fold, random_state=args.seed).split(df_train, outputs)

    rho_values = []

    for fold, (train_idx, valid_idx) in enumerate(kf):
        args.fold = fold
        model = get_model(args, config)
        train_X = [inputs[i][train_idx] for i in range(len(inputs))]
        train_Y = outputs[train_idx]

        valid_X = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_Y = outputs[valid_idx]

        val_rho_values = train_and_predict(model, train_data=(train_X, train_Y), valid_data=(valid_X, valid_Y),
                                           args=args)
        rho_values.append(np.array(val_rho_values))
        del model, train_X, train_Y, valid_X, valid_Y, val_rho_values
        [gc.collect() for _ in range(15)]

    train_rho_df = pd.DataFrame(np.array(rho_values), columns=[f'epoch-{x}' for x in range(args.epochs)])
    train_rho_df.to_csv(f'{args.output_dir}rho_values.csv', index=False)


if __name__ == '__main__':
    main()
