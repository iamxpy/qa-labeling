from config.bert_config import cfg
from src import *
import gc
import time
import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, RandomSampler
# import torch.utils.data as data
from torchvision import datasets, models, transforms
from sklearn.utils import shuffle
import random
from tqdm import tqdm
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from scipy.stats import spearmanr
import torch.nn as nn
import torch.nn.functional as F
import json
import math
from itertools import cycle


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(spearmanr(col_trues, col_pred).correlation)
    return np.mean(rhos)

cfg["EPOCH"] = 4
print(cfg)

BACK_STEP = 8
NFOLDS = cfg["NFOLDS"]
BATCH_SIZE = cfg["BATCH_SIZE"]
INFER_BATHC_SIZE = int(BATCH_SIZE * BACK_STEP)
swa_alpha = cfg["swa_alpha"]
# EPOCHS = cfg["EPOCHS"]
eval_every = cfg["EVAL_EVERY"]
LR = cfg["LR"]
MAX_SEQUENCE_LENGTH = cfg["MAX_SEQUENCE_LENGTH"]
Last_Layer = cfg["Last_Layer"]
Save_ckpt = cfg["Save_ckpt"]
Use_semi_supervised = cfg["Use_semi_supervised"]

if not os.path.exists(cfg["output"]):
    os.makedirs(cfg["output"])

save_dir = "./{}_ck_fold_{}_epoch_{}_LR_{}_layer_{}".format(cfg["pretrained_model"], NFOLDS, cfg["EPOCH"], LR, Last_Layer)

if os.path.exists(save_dir) and os.listdir(save_dir):
    for i in range(0, 100):
        tmp_save_dir = "{}_{}".format(save_dir, i)
        if not os.path.exists(tmp_save_dir):
            save_dir = tmp_save_dir
            break

print("save_dir {}".format(save_dir))
DATA_DIR = './google-quest-challenge'

SEED = 0
seed_everything(SEED)

print(config)

sub = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')

train = pd.read_csv(f'{DATA_DIR}/folds_trans_more.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

use_small_train = False
if use_small_train:
    train = train.loc[0:100, :]

if cfg["question"]:
    target_columns = sub.columns.values[1:22].tolist()
    output_categories = list(train.columns[11:32])
else:
    target_columns = sub.columns.values[22:].tolist()
    output_categories = list(train.columns[32:41])

input_categories = list(train.columns[[1, 2, 5]])
print('\noutput categories:\n\t', output_categories)
print(len(output_categories))
print("target_columns ", target_columns)
print('\ninput categories:\n\t', input_categories)


def predict_result(model, test_loader):
    test_preds = np.zeros((len(test), len(target_columns)))

    test_len = len(test)
    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            q_input_ids, q_input_masks, q_input_segments, qa_input_ids, qa_input_masks, qa_input_segments, _ = data
            if cfg["question"]:
                q_input_ids = q_input_ids.to(device)
                q_input_masks = q_input_masks.to(device)
                q_input_segments = q_input_segments.to(device)
                pred = model(q_input_ids, q_input_masks, q_input_segments)

            else:
                qa_input_ids = qa_input_ids.to(device)
                qa_input_masks = qa_input_masks.to(device)
                qa_input_segments = qa_input_segments.to(device)
                pred = model(qa_input_ids, qa_input_masks, qa_input_segments)

            test_preds[
            idx * INFER_BATHC_SIZE: min((idx + 1) * INFER_BATHC_SIZE, test_len)] = pred.detach().cpu().numpy()

    return test_preds


def validation(model, val_loader, loss_fn):
    sum_val_loss = 0.0
    model.eval()

    valid_preds = []
    true_label = []

    with torch.no_grad():
        for j, data in enumerate(val_loader):
            q_input_ids, q_input_masks, q_input_segments, qa_input_ids, qa_input_masks, qa_input_segments, labels = data
            if cfg["question"]:
                q_input_ids = q_input_ids.to(device)
                q_input_masks = q_input_masks.to(device)
                q_input_segments = q_input_segments.to(device)
                pred = model(q_input_ids, q_input_masks, q_input_segments)

            else:
                qa_input_ids = qa_input_ids.to(device)
                qa_input_masks = qa_input_masks.to(device)
                qa_input_segments = qa_input_segments.to(device)
                pred = model(qa_input_ids, qa_input_masks, qa_input_segments)

            labels = labels.to(device)
            loss = loss_fn(pred, labels.float())
            sum_val_loss += loss.item()

            valid_preds.append(pred.cpu().detach().numpy())
            true_label.append(labels.cpu().detach().numpy())

    true_label = np.concatenate(true_label)
    valid_preds = np.concatenate(valid_preds)
    score = compute_spearmanr(true_label, valid_preds)

    return sum_val_loss / len(val_loader), score


def optim_parm_collect(model, LR, pre_model):
    if "albert" in cfg["pretrained_model"]:
        optimizer_grouped_parameters = [
            {'params': model.bert.embeddings.parameters(), 'lr': LR * (0.95 ** cfg["NUM_LAYERS"])},
            {'params': model.bert.encoder.embedding_hidden_mapping_in.parameters(), 'lr': LR * (0.95 ** cfg["NUM_LAYERS"])},
            {'params': model.head.parameters(), 'lr': LR},
            {'params': model.bert.encoder.albert_layer_groups.parameters(), 'lr': LR * (0.9 ** cfg["NUM_LAYERS"])},
        ]

    elif "bert" in cfg["pretrained_model"]:
        optimizer_grouped_parameters = [
            {'params': model.bert.embeddings.parameters(), 'lr': LR * (0.95 ** cfg["NUM_LAYERS"])},
            {'params': model.bert_qa.embeddings.parameters(), 'lr': LR * (0.95 ** cfg["NUM_LAYERS"])},
            {'params': model.head1.parameters(), 'lr': LR},
            {'params': model.head2.parameters(), 'lr': LR},
            {'params': model.bert.pooler.parameters(), 'lr': LR},
            {'params': model.bert_qa.pooler.parameters(), 'lr': LR}
        ]

        for layer in range(cfg["NUM_LAYERS"]):
            optimizer_grouped_parameters.append(
                {'params': model.bert.encoder.layer.__getattr__(
                    '%d' % (cfg["NUM_LAYERS"] - 1 - layer)).parameters(),
                 'lr': LR * (0.95 ** layer)},
            )
            optimizer_grouped_parameters.append(
                {'params': model.bert_qa.encoder.layer.__getattr__(
                    '%d' % (cfg["NUM_LAYERS"] - 1 - layer)).parameters(),
                 'lr': LR * (0.95 ** layer)},
            )

    return optimizer_grouped_parameters


def train_model(tokenizer, train, save_dir, test_loader):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(cfg["output"]):
        os.makedirs(cfg["output"])
    print("train_model ", train.shape)
    average_spearman = 0
    average_swa_score = 0

    results = np.zeros((len(test), len(target_columns)))
    swa_results = np.zeros((len(test), len(target_columns)))

    for cv in range(NFOLDS):
        print(f'fold {cv + 1}')

        train_df = train[train['fold'] != cv]
        train_df_more = train_df.copy(deep=True)

        if cfg["upsample"] != None:
            for i, row in train_df.iterrows():
                if row['question_type_spelling'] > 0:
                    a = train_df.loc[i]
                    d = pd.DataFrame(a).T
                    train_df_more = train_df_more.append([d] * 15 * int(cfg["upsample"]))
                if row['question_not_really_a_question'] > 0:
                    a = train_df.loc[i]
                    d = pd.DataFrame(a).T
                    train_df_more = train_df_more.append([d] * 4 * int(cfg["upsample"]))
                if row['question_type_consequence'] > 0:
                    a = train_df.loc[i]
                    d = pd.DataFrame(a).T
                    train_df_more = train_df_more.append([d] * 2 * int(cfg["upsample"]))
        else:
            print("no upsample")

        del train_df
        gc.collect()

        eval_df = train[train['fold'] == cv]

        x_train = convert_lines(tokenizer, train_df_more['question_title'], train_df_more['question_body'],
                                train_df_more['answer'])
        y_train = train_df_more.loc[:, output_categories].values.astype(np.float)

        print(x_train[0].shape)
        x_eval = convert_lines(tokenizer, eval_df['question_title'], eval_df['question_body'], eval_df['answer'])
        y_eval = eval_df.loc[:, output_categories].values.astype(np.float)

        train_dataset = QuestDataset(x_train, y_train)
        train_sampler = RandomSampler(train_dataset)

        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, pin_memory=True)

        val_loader = DataLoader(QuestDataset(x_eval, y_eval), batch_size=INFER_BATHC_SIZE, shuffle=False,
                                pin_memory=True)

        STEPS = cfg["EPOCH"] * math.ceil(len(train_loader))

        print("tot steps {}".format(STEPS))
        if cfg["question"]:
            model = ALBertForQuest()
            swa_model = ALBertForQuest()
        else:
            model = ALBertForAnswer()
            swa_model = ALBertForAnswer()

        for param in swa_model.parameters():
            param.detach_()

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        print(
            'iter epoch | valid_loss  spearmanr | swa_valid_loss  swa_score | train_loss |  val time | time | lr  ')
        print(
            '------------------------------------------------------------------------------------------------------------------------------------------------')

        model = model.to(device)
        swa_model = swa_model.to(device)

        loss_fn = torch.nn.BCELoss()

        LR = 7e-5
        optimizer_grouped_parameters = optim_parm_collect(model, LR, pre_model)
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=LR)

        eval_every = 800
        best_score = 0.0
        swa_best_score = 0.0
        sum_loss = 0.0

        n_avg = 0

        epoch_step = len(train_loader)

        train_loader = cycle(train_loader)
        start_time = time.time()

        for step in range(STEPS):
            data = next(train_loader)

            if 0 <= (step / epoch_step - 2) <= 0.1:
                LR = 3e-6
                optimizer_grouped_parameters = optim_parm_collect(model, LR, pre_model)
                optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=LR)

            q_input_ids, q_input_masks, q_input_segments, qa_input_ids, qa_input_masks, qa_input_segments, labels = data
            if cfg["question"]:
                q_input_ids = q_input_ids.to(device)
                q_input_masks = q_input_masks.to(device)
                q_input_segments = q_input_segments.to(device)
                pred = model(q_input_ids, q_input_masks, q_input_segments)

            else:
                qa_input_ids = qa_input_ids.to(device)
                qa_input_masks = qa_input_masks.to(device)
                qa_input_segments = qa_input_segments.to(device)
                pred = model(qa_input_ids, qa_input_masks, qa_input_segments)

            labels = labels.to(device)

            loss = loss_fn(pred, labels.float())

            sum_loss += loss.item()

            loss.backward()
            if (step + 1) % BACK_STEP == 0:
                optimizer.step()
                optimizer.zero_grad()

                # 从第1.5个epoch开始
                if 1.5 <= step / epoch_step:
                    virtual_decay = 1 / float(n_avg + 1)
                    for swa_param, param in zip(swa_model.parameters(), model.parameters()):
                        swa_param.data.mul_(1.0 - virtual_decay).add_(virtual_decay, param.data)
                    n_avg += 1

            if (step + 1) % eval_every == 0:
                tmp_time = time.time()
                val_loss, score = validation(model, val_loader, loss_fn)

                if step / epoch_step >= 1.5:
                    swa_val_loss, swa_score = validation(swa_model, val_loader, loss_fn)
                else:
                    swa_val_loss, swa_score = 0, 0

                val_time = time.time() - tmp_time

                if score > best_score:  # 记录未经过任何处理的best score
                    best_score = score

                    if Save_ckpt:
                        model_path = "{}/{}_bert_fold_{}.pth".format(save_dir, cfg["output"], cv)
                        torch.save(model.state_dict(), model_path)

                if swa_score > swa_best_score:  # 记录未经过任何处理的best score
                    swa_best_score = swa_score

                    if Save_ckpt:
                        swa_model_path = "{}/{}_swa_bert_fold_{}.pth".format(save_dir, cfg["output"], cv)
                        torch.save(swa_model.state_dict(), swa_model_path)

                elapsed_time = time.time() - start_time

                print(
                    '{:.1f}k  {:.1f}  |  {:.4f}  {:.4f} | {:.4f}  {:.4f} | {:.4f}  | {:.1f}  {:.1f}| {:.7f}'.format(
                        step / 1000, step / epoch_step, val_loss, score, swa_val_loss, swa_score, sum_loss / (step + 1), elapsed_time, val_time,
                        optimizer.param_groups[0]['lr']))

                model.train()

        infer_time = time.time()
        model.load_state_dict(torch.load("{}/{}_bert_fold_{}.pth".format(save_dir, cfg["output"], cv)))
        swa_model.load_state_dict(torch.load("{}/{}_swa_bert_fold_{}.pth".format(save_dir, cfg["output"], cv)))

        result = predict_result(model, test_loader)
        swa_result = predict_result(swa_model, test_loader)
        print("infer time ", time.time() - infer_time)

        if True:
            sub[target_columns] = result
            sub.to_csv("{}/fold_{}.csv".format(cfg["output"], cv), index=False)

            sub[target_columns] = swa_result
            sub.to_csv("{}/swa_fold_{}.csv".format(cfg["output"], cv), index=False)

        results += result
        swa_results += swa_result

        average_spearman += best_score
        average_swa_score += swa_best_score

        torch.cuda.empty_cache()

    print("*********************average******************")
    average_spearman /= NFOLDS
    average_swa_score /= NFOLDS
    print("{} {}".format(average_spearman, average_swa_score))

    return results / NFOLDS, swa_results / NFOLDS


x_test = convert_lines(tokenizer, test['question_title'], test['question_body'], test['answer'])
test_loader = torch.utils.data.DataLoader(QuestDataset(x_test),
                                          batch_size=INFER_BATHC_SIZE, shuffle=False)

if True:
    if cfg["question"]:
        cfg["output"] += "_question"
    else:
        cfg["output"] += "_answer"

    preds = train_model(tokenizer, train, save_dir, test_loader)

    sub[target_columns] = preds
    sub.to_csv("{}/result.csv".format(cfg["output"]), index=False)
