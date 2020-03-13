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
from modeling_bert import BertForSequenceClassification_gru, BertConfig
from optimization import AdamW, WarmupLinearSchedule
from tokenization_bert import BertTokenizer
from itertools import cycle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification_gru, BertTokenizer),
}
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 choices_features,
                 label
                 ):
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def read_examples(df: pd.DataFrame, feature_labels: list, label_columns: list = None):
    columns = list(feature_labels)
    if label_columns is not None:
        columns += label_columns
    examples = []
    for val in df[columns].values:
        if label_columns is not None:
            examples.append(InputExample(text_a=val[0], text_b=val[1], label=val[2:]))
        else:
            examples.append(InputExample(text_a=val[0], text_b=val[1], label=None))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, split_num, verbose=False):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for example_index, example in enumerate(examples):

        short_tokens = tokenizer.tokenize(example.text_a)
        long_tokens = tokenizer.tokenize(example.text_b)

        skip_len = len(long_tokens) / split_num
        choices_features = []
        for i in range(split_num):
            chosen_long_tokens = long_tokens[int(i * skip_len):int((i + 1) * skip_len)]
            _truncate_seq_pair(chosen_long_tokens, short_tokens, max_seq_length - 3)
            tokens = ["[CLS]"] + short_tokens + ["[SEP]"] + chosen_long_tokens + ["[SEP]"]
            segment_ids = [0] * (len(short_tokens) + 2) + [1] * (len(chosen_long_tokens) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_ids += ([0] * padding_length)
            choices_features.append((tokens, input_ids, input_mask, segment_ids))

            label = example.label
            if verbose and example_index < 1:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))
                logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                choices_features=choices_features,
                label=label
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task", type=str, required=True,
                        help='Modeling for "question" or "answer"')
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int, help="")
    parser.add_argument("--fold", required=True, type=int, help="Run on which fold of the dataset")
    parser.add_argument("--lstm_hidden_size", default=300, type=int, help="")
    parser.add_argument("--lstm_layers", default=2, type=int, help="")
    parser.add_argument("--lstm_dropout", default=0.5, type=float, help="")

    parser.add_argument("--train_steps", default=-1, type=int, help="")
    parser.add_argument("--report_steps", default=-1, type=int, help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--split_num", default=3, type=int,
                        help="text split")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending "
                             "and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--freeze", default=0, type=int, required=False,
                        help="Freeze parameters in bert.")
    parser.add_argument("--not_do_eval_steps", default=0.35, type=float,
                        help="Use a ratio of train_steps to specify a early stage where we dont do eval.")
    args = parser.parse_args()

    args.data_dir = os.path.join(args.data_dir, f'fold-{args.fold}')
    args.output_dir = os.path.join(args.output_dir, f'fold-{args.fold}')

    # Setup GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # todo delete head
    # df_train = pd.read_csv(os.path.join(args.data_dir, 'train.csv')).head(50)
    # df_dev = pd.read_csv(os.path.join(args.data_dir, 'dev.csv')).head(20)
    # df_test = pd.read_csv(os.path.join(args.data_dir, 'test.csv')).head(30)
    df_train = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    df_dev = pd.read_csv(os.path.join(args.data_dir, 'dev.csv'))
    df_test = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))
    print('train shape =', df_train.shape)
    print('test shape =', df_test.shape)

    # todo 写一个预处理脚本，清洗文本以及将train切分为K折
    # Read data
    if args.task == "question":
        input_categories = list(df_train.columns[[1, 2]])
        output_categories = list(df_train.columns[11:32])  # 21 categories
        num_labels = 21
    elif args.task == "answer":
        input_categories = list(df_train.columns[[1, 5]])
        output_categories = list(df_train.columns[32:])  # 9 categories
        num_labels = 9
    else:
        raise ValueError("Task must be one of 'question' and 'answer'")
    print('\ninput categories:\n\t', input_categories)
    print('\noutput categories:\n\t', output_categories)

    # Prepare model
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    config.model_name_or_path=args.model_name_or_path
    config.lstm_dropout=args.lstm_dropout
    config.lstm_hidden_size=args.lstm_hidden_size
    config.lstm_layers=args.lstm_layers
    config.device=args.device

    # todo 修改网络架构之后需要修改model类
    model = BertForSequenceClassification_gru(config)
    model.to(device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    if args.do_train:

        # Prepare data loader

        train_examples = read_examples(df_train, feature_labels=input_categories,
                                       label_columns=output_categories)
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, args.split_num, verbose=True)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.float32)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        # Prepare optimizer
        num_train_optimization_steps = args.train_steps
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # 一般只对weight做正则化而不管bias，至于LayerNorm层的参数为啥不用正则化，还不知道
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                         t_total=args.train_steps // args.gradient_accumulation_steps)

        print("\n" + "*" * 30 + " Running training " + "*" * 30 + "\n")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        global_step = 0
        tr_loss = 0
        best_rho = 0
        nb_tr_steps = 0

        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)  # cycle('ABCD') --> A B C D A B C D A B C D ...
        model.train()  # Sets the module in training mode

        # num_train_optimization_steps是一共要进行的前向+反向（但不更新参数）的次数，step是已经进行的次数
        # 如果设置了gradient_accumulation_steps，则global_step=step//gradient_accumulation_steps
        # 即global_step是已经更新参数的次数（每gradient_accumulation_steps次前向+反向就更新1次）
        for step in bar:  # step是传给网络的batch数，但如果设置了累积梯度计算，实际更新的步数是变量global_step而不是这里的step
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss, _ = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                            labels=label_ids)  # 训练集上得到的logits直接丢弃，不需要计算指标
            del input_ids, input_mask, segment_ids, label_ids
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()  # todo 用tensorboard记录这些指标
            # tr_loss*args.gradient_accumulation_steps是真实的batch累积loss，除以training的batch数得到平均batch loss
            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)

            bar.set_description("loss {}".format(round(train_loss, 4)))
            nb_tr_steps += 1  # 记录training的batch个数（不就是等于step吗？）
            loss.backward()  # 反向传播计算梯度，但是每gradient_accumulation_steps个batch再更新

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # 利用累积了多个batch的梯度来更新参数
                scheduler.step()
                optimizer.zero_grad()  # 清空梯度
                global_step += 1  # 虚拟批次数加一

            # eval_steps指的是每多少个全局步数global_step就报告一次当前训练结果
            if (step + 1) % (args.eval_steps * args.gradient_accumulation_steps) == 0:
                tr_loss = 0
                nb_tr_steps = 0
                print("\n" + "*" * 30 + " Report result " + "*" * 30 + "\n")
                logger.info("  %s = %s", 'global_step', str(global_step))
                logger.info("  %s = %s", 'train loss', str(train_loss))

            # not_do_eval_steps是指前多少比例的num_train_optimization_steps（前向+反向但不更新参数的次数）不进行evel
            if args.do_eval and step > num_train_optimization_steps * args.not_do_eval_steps and (step + 1) % (
                    args.eval_steps * args.gradient_accumulation_steps) == 0:
                predictions = []
                gold_labels = []
                eval_examples = read_examples(df_dev, feature_labels=input_categories,
                                              label_columns=output_categories)
                eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,
                                                             args.split_num)
                all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
                all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
                all_label = torch.tensor([f.label for f in eval_features], dtype=torch.float32)

                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

                print("\n" + "*" * 30 + " Running evaluation " + "*" * 30 + "\n")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                eval_loss, eval_rho, nb_eval_steps, nb_eval_examples = 0, 0, 0, 0
                for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

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

                # todo 用tensorboard记录指标
                result = {'eval_loss': eval_loss,
                          'eval_rho': eval_rho,
                          'global_step': global_step,
                          'loss': train_loss}

                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

                with open(output_eval_file, "a") as writer:
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
                    writer.write('*' * 80)
                    writer.write('\n')
                if eval_rho > best_rho:
                    print("=" * 80)
                    print("Best rho:", eval_rho)
                    print("Saving Model......")
                    best_rho = eval_rho
                    # Save a trained model, todo 看下torch怎么保存和加载模型
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")  # todo 用对应一些参数来保存模型名
                    torch.save(model_to_save.state_dict(), output_model_file)
                    # 保存最好的预测结果
                    df_dev.loc[:, output_categories] = predictions
                    df_dev[['qa_id'] + output_categories].to_csv(
                        os.path.join(args.output_dir, f"sub_{args.task}_dev.csv"),
                        index=False)
                    print("=" * 80)
                else:
                    print("=" * 80)

                model.train()

    if args.do_test:
        del model
        gc.collect()
        args.do_train = False  # 加载最好的eval结果最好的模型用于预测dev集和test集的结果
        model = BertForSequenceClassification_gru.from_pretrained(os.path.join(args.output_dir, "pytorch_model.bin"),
                                                                  args,
                                                                  config=config)  # todo 修改网络架构之后需要修改这里
        model.to(device)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        predictions = []
        test_examples = read_examples(df_test, feature_labels=input_categories)
        test_features = convert_examples_to_features(test_examples, tokenizer, args.max_seq_length, args.split_num)
        all_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        model.eval()
        for input_ids, input_mask, segment_ids in test_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                batch_logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
            predictions.append(torch.sigmoid(batch_logits).cpu().numpy())
        predictions = np.concatenate(predictions, 0)
        df_predictions = pd.DataFrame(predictions, columns=output_categories)
        df_test = pd.concat([df_test, df_predictions], axis=1)
        df_test[['qa_id'] + output_categories].to_csv(os.path.join(args.output_dir, f"sub_{args.task}_test.csv"),
                                                      index=False)


if __name__ == "__main__":
    main()
