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
from modeling_bert import BertConfig, BertForSequenceClassification_linear_multi
from bert_utils import AdamW, WarmupLinearSchedule
from bert_utils import BertTokenizer
from bert_utils import compute_3sen_input_arrays, compute_1sen_input_arrays, compute_output_arrays
from stratifiers import MultilabelStratifiedKFold
from math import ceil, floor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification_linear_multi, BertTokenizer),
}
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

logger = logging.getLogger(__name__)


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
    q_ids, q_masks, q_segs, qa_ids, qa_masks, qa_segs = X
    eval_labels = Y
    # 每个epoch的结尾进行一次Evaluation
    predictions = []
    gold_labels = []
    q_ids = torch.tensor(q_ids, dtype=torch.long)
    q_masks = torch.tensor(q_masks, dtype=torch.long)
    q_segs = torch.tensor(q_segs, dtype=torch.long)
    qa_ids = torch.tensor(qa_ids, dtype=torch.long)
    qa_masks = torch.tensor(qa_masks, dtype=torch.long)
    qa_segs = torch.tensor(qa_segs, dtype=torch.long)
    eval_labels_tensor = torch.tensor(eval_labels, dtype=torch.float32)

    eval_data = TensorDataset(q_ids, q_masks, q_segs, qa_ids, qa_masks, qa_segs,
                              eval_labels_tensor)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss, eval_rho, nb_eval_steps, nb_eval_examples = 0, 0, 0, 0
    for b_q_ids, b_q_masks, b_q_segs, b_qa_ids, b_qa_masks, b_qa_segs, b_labels in eval_dataloader:
        b_q_ids = b_q_ids.to(args.device)
        b_q_masks = b_q_masks.to(args.device)
        b_q_segs = b_q_segs.to(args.device)
        b_qa_ids = b_qa_ids.to(args.device)
        b_qa_masks = b_qa_masks.to(args.device)
        b_qa_segs = b_qa_segs.to(args.device)
        b_labels = b_labels.to(args.device)

        with torch.no_grad():
            batch_eval_loss, batch_logits = model(b_q_ids, b_q_masks, b_q_segs, b_qa_ids, b_qa_masks, b_qa_segs,
                                                  b_labels)
        batch_preds = torch.sigmoid(batch_logits).cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # 将所有batch得到的预测标签拼接起来, shape: [eval_batch_size,num_class]
        predictions.append(batch_preds)
        # 将所有batch的真实标签拼接起来, shape: [eval_batch_size,num_class]
        gold_labels.append(label_ids)
        eval_loss += batch_eval_loss.mean().item()
        nb_eval_examples += b_q_ids.size(0)
        nb_eval_steps += 1
        del b_q_ids, b_q_masks, b_q_segs, b_qa_ids, b_qa_masks, b_qa_segs, b_labels
        [gc.collect() for _ in range(15)]

    gold_labels = np.concatenate(gold_labels, 0)
    predictions = np.concatenate(predictions, 0)
    eval_loss = eval_loss / nb_eval_steps
    eval_rho = compute_spearmanr(gold_labels, predictions)
    return predictions, eval_loss, eval_rho


def train_and_predict(model, train_data, valid_data, args):
    # Training data
    question_ids, question_masks, question_segments, qa_ids, qa_masks, qa_segments = train_data[0]
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
    t_total = (args.epochs * int(ceil(qa_ids.shape[0] / args.train_batch_size))) // args.gradient_accumulation_steps
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                     t_total=t_total)

    tr_loss = 0  # 一个epoch的累积loss，用于计算平均loss
    nb_tr_steps = 0  # 一个epoch内已经前向计算的batch数
    val_rhos = []  # 记录validation的rho，用于后期voting

    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}/{args.epochs}")
        # Prepare data loader
        question_ids_tensor = torch.tensor(question_ids, dtype=torch.long)
        question_mask_tensor = torch.tensor(question_masks, dtype=torch.long)
        question_segments_tensor = torch.tensor(question_segments, dtype=torch.long)
        qa_ids_tensor = torch.tensor(qa_ids, dtype=torch.long)
        qa_mask_tensor = torch.tensor(qa_masks, dtype=torch.long)
        qa_segments_tensor = torch.tensor(qa_segments, dtype=torch.long)

        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)

        train_data = TensorDataset(question_ids_tensor, question_mask_tensor, question_segments_tensor,
                                   qa_ids_tensor, qa_mask_tensor, qa_segments_tensor,
                                   train_labels_tensor)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        bar = tqdm(total=qa_ids.shape[0], ncols=100)
        model.train()  # Sets the module in training mode
        for batch_q_ids, batch_q_masks, batch_q_segs, batch_qa_ids, batch_qa_masks, batch_qa_segs, label_ids in train_dataloader:
            actual_batch_size = batch_q_ids.shape[0]  # 用于更新tqdm, 最后一个batch大小可能不等于batch_size
            batch_q_ids = batch_q_ids.to(args.device)
            batch_q_masks = batch_q_masks.to(args.device)
            batch_q_segs = batch_q_segs.to(args.device)
            batch_qa_ids = batch_qa_ids.to(args.device)
            batch_qa_masks = batch_qa_masks.to(args.device)
            batch_qa_segs = batch_qa_segs.to(args.device)
            label_ids = label_ids.to(args.device)
            loss, _ = model(batch_q_ids, batch_q_masks, batch_q_segs, batch_qa_ids, batch_qa_masks, batch_qa_segs,
                            labels=label_ids)  # 训练集上得到的logits直接丢弃，不需要计算指标
            del batch_q_ids, batch_q_masks, batch_q_segs, batch_qa_ids, batch_qa_masks, batch_qa_segs, label_ids
            [gc.collect() for _ in range(15)]
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()  # todo 用tensorboard记录这些指标
            # tr_loss是真实的batch累积loss，除以training的batch数得到平均batch loss
            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)

            bar.set_description(f"loss {round(train_loss, 4)}")
            nb_tr_steps += 1  # 记录一个epoch内已经前向传播的batch个数
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # 利用累积了多个batch的梯度来更新参数
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
    model = BertForSequenceClassification_linear_multi(config)
    # model = BertForSequenceClassification_linear_multi.from_pretrained(config.model_name_or_path, config=config)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model


def get_tokenizer_and_config(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    config = BertConfig.from_pretrained(args.model_name_or_path)
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
    args.epochs = 10
    args.eval_steps = 3
    args.max_seq_length = 512
    args.split_num = 1
    args.lstm_hidden_size = 512
    args.lstm_layers = 1
    args.lstm_dropout = 0.1
    args.dropout = 0.2
    args.per_gpu_train_batch_size = 8
    args.gradient_accumulation_steps = 4
    args.warmup_steps = 0
    args.per_gpu_eval_batch_size = 32
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
    # todo delete head
    df_train = pd.read_csv(args.data_dir + 'train.csv').head(100)
    print('train shape =', df_train.shape)
    # Read data
    output_categories = list(df_train.columns[11:])  # 30 categories
    question_input_categories = list(df_train.columns[[1, 2]])
    qa_input_categories = list(df_train.columns[[1, 2, 5]])

    tokenizer, config = get_tokenizer_and_config(args)

    # 需要被模型内部使用的参数在这里加入
    config.lstm_hidden_size = args.lstm_hidden_size
    config.lstm_layers = args.lstm_layers
    config.lstm_dropout = args.lstm_dropout
    config.split_num = args.split_num
    config.dropout = args.dropout
    config.device = args.device
    config.model_name_or_path = args.model_name_or_path

    inputs = []
    inputs += compute_1sen_input_arrays(df_train, question_input_categories, tokenizer, args.max_seq_length)
    inputs += compute_3sen_input_arrays(df_train, qa_input_categories, tokenizer, args.max_seq_length)
    outputs = compute_output_arrays(df_train, output_categories)

    kf = MultilabelStratifiedKFold(n_splits=args.num_fold).split(df_train, outputs)

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
