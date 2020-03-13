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


print(cfg)

NFOLDS = cfg["NFOLDS"]
BATCH_SIZE = cfg["BATCH_SIZE"]
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

save_dir = "./ck_fold_{}_epoch_{}_LR_{}_layer_{}".format(NFOLDS, cfg["EPOCH"], LR, Last_Layer)

if os.path.exists(save_dir) and os.listdir(save_dir):
    for i in range(0, 1000):
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
target_columns = sub.columns.values[1:].tolist()

train = pd.read_csv(f'{DATA_DIR}/folds_trans_more.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

use_small_train = False
if use_small_train:
    train = train.loc[0:100, :]

output_categories = list(train.columns[11:41])
input_categories = list(train.columns[[1, 2, 5]])
print('\noutput categories:\n\t', output_categories)
print(len(output_categories))
print('\ninput categories:\n\t', input_categories)

# 有些列绝大部分时候加上任何后处理都只会降低分数，所以禁用对这些列禁用后处理，此处的数字是列在output_category的下标
ban_list = [0, 1, 4, 6, 17, 18] + list(range(20, 30))
# union后处理方案是指结合各种后处理方案表现较好的列，不过经过分析post方法没有优势，所以实际上只使用了other和align的结果
union_from_other = [3, 7, 8, 9, 10, 11, 13, 16]
union_from_align = [2, 5, 12, 14, 15, 19]


def predict_result(model, test_loader):
    test_preds = np.zeros((len(test), len(target_columns)))

    test_len = len(test)
    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            q_input_ids, q_input_masks, q_input_segments, qa_input_ids, qa_input_masks, qa_input_segments, _ = data
            q_input_ids = q_input_ids.to(device)
            q_input_masks = q_input_masks.to(device)
            q_input_segments = q_input_segments.to(device)

            qa_input_ids = qa_input_ids.to(device)
            qa_input_masks = qa_input_masks.to(device)
            qa_input_segments = qa_input_segments.to(device)

            pred = model(q_input_ids, q_input_masks, q_input_segments, qa_input_ids, qa_input_masks,
                         qa_input_segments)

            test_preds[idx * BATCH_SIZE: min((idx + 1) * BATCH_SIZE, test_len)] = pred.detach().cpu().numpy()

    test_preds_new = test_preds.copy()
    align_preds = test_preds.copy()

    post_test_preds = deal_result(test_preds, ban_list)
    other_test_preds = post_deal_result(test_preds_new, ban_list)
    align_test_preds = align(align_preds, ban_list)
    return test_preds, post_test_preds, other_test_preds, align_test_preds


def cal(arr1, arr2):
    return np.array([compute_spearmanr(arr1[:, i].reshape(-1, 1), arr2[:, i].reshape(-1, 1)) for i in range(30)])


def validation(model, val_loader, loss_fn, post_diffs, other_diffs, align_diffs, fold):
    sum_val_loss = 0.0
    model.eval()

    valid_preds = []
    true_label = []

    with torch.no_grad():
        for j, data in enumerate(val_loader):
            q_input_ids, q_input_masks, q_input_segments, qa_input_ids, qa_input_masks, qa_input_segments, labels = data
            q_input_ids = q_input_ids.to(device)
            q_input_masks = q_input_masks.to(device)
            q_input_segments = q_input_segments.to(device)

            qa_input_ids = qa_input_ids.to(device)
            qa_input_masks = qa_input_masks.to(device)
            qa_input_segments = qa_input_segments.to(device)

            labels = labels.to(device)

            pred = model(q_input_ids, q_input_masks, q_input_segments, qa_input_ids, qa_input_masks,
                         qa_input_segments)

            loss = loss_fn(pred, labels.float())
            sum_val_loss += loss.item()

            valid_preds.append(pred.cpu().detach().numpy())
            true_label.append(labels.cpu().detach().numpy())

    true_label = np.concatenate(true_label)
    valid_preds = np.concatenate(valid_preds)

    valid_preds_post = valid_preds.copy()
    valid_preds_other = valid_preds.copy()
    valid_preds_align = valid_preds.copy()
    valid_preds_union = valid_preds.copy()

    score = compute_spearmanr(true_label, valid_preds)
    ori_scores = cal(true_label, valid_preds)

    # post without ban socres
    valid_preds_post = deal_result(valid_preds_post)
    post_scores = cal(true_label, valid_preds_post)
    diff = pd.DataFrame(post_scores - ori_scores).T
    diff.columns = output_categories
    post_diffs = post_diffs.append(diff)
    post_diffs.to_csv("{}/fold_{}_post_diffs.csv".format(cfg["output"], fold), index=False)
    # post with ban
    for c in ban_list:
        valid_preds_post[:, c] = valid_preds[:, c]
    post_score = compute_spearmanr(true_label, valid_preds_post)

    # other without ban scores
    valid_preds_other = post_deal_result(valid_preds_other)
    other_scores = cal(true_label, valid_preds_other)
    diff = pd.DataFrame(other_scores - ori_scores).T
    diff.columns = output_categories
    other_diffs = other_diffs.append(diff)
    other_diffs.to_csv("{}/fold_{}_other_diffs.csv".format(cfg["output"], fold), index=False)
    # other with ban
    for c in ban_list:
        valid_preds_other[:, c] = valid_preds[:, c]
    other_score = compute_spearmanr(true_label, valid_preds_other)
    # union form other
    for c in union_from_other:
        valid_preds_union[:, c] = valid_preds_other[:, c]

    # align without ban scores
    valid_preds_align = align(valid_preds_align)
    align_scores = cal(true_label, valid_preds_align)
    diff = pd.DataFrame(align_scores - ori_scores).T
    diff.columns = output_categories
    align_diffs = align_diffs.append(diff)
    align_diffs.to_csv("{}/fold_{}_align_diffs.csv".format(cfg["output"], fold), index=False)
    # align with ban
    for c in ban_list:
        valid_preds_align[:, c] = valid_preds[:, c]
    align_score = compute_spearmanr(true_label, valid_preds_align)
    # union form align
    for c in union_from_align:
        valid_preds_union[:, c] = valid_preds_align[:, c]

    # union
    union_score = compute_spearmanr(true_label, valid_preds_union)

    return sum_val_loss / len(
        val_loader), score, post_score, other_score, align_score, union_score, post_diffs, other_diffs, align_diffs

def optim_parm_collect(model, LR, pre_model):
    if torch.cuda.device_count() > 1:
        optimizer_grouped_parameters = [
            {'params': model.module.bert.embeddings.parameters(), 'lr': LR * (0.95 ** cfg["NUM_LAYERS"])},
            {'params': model.module.bert_qa.embeddings.parameters(), 'lr': LR * (0.95 ** cfg["NUM_LAYERS"])},
            {'params': model.module.head1.parameters(), 'lr': LR},
            {'params': model.module.head2.parameters(), 'lr': LR},
            {'params': model.module.bert.pooler.parameters(), 'lr': LR},
            {'params': model.module.bert_qa.pooler.parameters(), 'lr': LR}

        ]

        for layer in range(cfg["NUM_LAYERS"]):
            optimizer_grouped_parameters.append(
                {'params': model.module.bert.encoder.layer.__getattr__(
                    '%d' % (cfg["NUM_LAYERS"] - 1 - layer)).parameters(),
                 'lr': LR * (0.95 ** layer)},
            )
            optimizer_grouped_parameters.append(
                {'params': model.module.bert_qa.encoder.layer.__getattr__(
                    '%d' % (cfg["NUM_LAYERS"] - 1 - layer)).parameters(),
                 'lr': LR * (0.95 ** layer)},
            )
    else:
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
    print("train_model ", train.shape)
    average_spearman = 0
    average_post_score = 0
    average_other_score = 0
    average_align_score = 0
    average_swa_score = 0
    average_swa_post_score = 0
    average_swa_other_score = 0
    average_swa_align_score = 0

    results = np.zeros((len(test), len(target_columns)))
    other_results = np.zeros((len(test), len(target_columns)))
    post_results = np.zeros((len(test), len(target_columns)))
    align_results = np.zeros((len(test), len(target_columns)))

    swa_results = np.zeros((len(test), len(target_columns)))
    swa_other_results = np.zeros((len(test), len(target_columns)))
    swa_post_results = np.zeros((len(test), len(target_columns)))
    swa_align_results = np.zeros((len(test), len(target_columns)))

    for cv in range(NFOLDS):
        print(f'fold {cv + 1}')

        post_diffs = pd.DataFrame(columns=output_categories)
        other_diffs = pd.DataFrame(columns=output_categories)
        align_diffs = pd.DataFrame(columns=output_categories)

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

        val_loader = DataLoader(QuestDataset(x_eval, y_eval), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        STEPS = cfg["EPOCH"] * math.ceil(len(train_loader))

        print("tot steps {}".format(STEPS))
        model = BertForQuest()
        swa_model = BertForQuest()
        for param in swa_model.parameters():
            param.detach_()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            swa_model = nn.DataParallel(swa_model)

        print(
            'iter epoch | valid_loss  spearmanr post_score other_score align_score union_score| swa_loss swa_score swa_post_score swa_other_score swa_align_score swa_union_score|  train_loss |  time          ')
        print(
            '------------------------------------------------------------------------------------------------------------------------------------------------')

        model = model.to(device)
        swa_model = swa_model.to(device)

        loss_fn = torch.nn.BCELoss()

        LR = 3e-5
        optimizer_grouped_parameters = optim_parm_collect(model, LR, pre_model)
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=LR)

        best_score = 0.0
        swa_best_score = 0.0
        corr_post_score = 0.0
        corr_other_score = 0.0
        corr_align_score = 0.0
        corr_swa_post_score = 0.0
        corr_swa_other_score = 0.0
        corr_swa_align_score = 0.0

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

            optimizer.zero_grad()
            q_input_ids, q_input_masks, q_input_segments, qa_input_ids, qa_input_masks, qa_input_segments, labels = data
            q_input_ids = q_input_ids.to(device)
            q_input_masks = q_input_masks.to(device)
            q_input_segments = q_input_segments.to(device)

            qa_input_ids = qa_input_ids.to(device)
            qa_input_masks = qa_input_masks.to(device)
            qa_input_segments = qa_input_segments.to(device)

            labels = labels.to(device)

            pred = model(q_input_ids, q_input_masks, q_input_segments, qa_input_ids, qa_input_masks,
                         qa_input_segments)

            loss = loss_fn(pred, labels.float())

            sum_loss += loss.item()

            loss.backward()
            optimizer.step()

            # 从第1.5个epoch开始
            if 1.5 <= step / epoch_step <= 3:
                virtual_decay = 1 / float(n_avg + 1)
                for swa_param, param in zip(swa_model.parameters(), model.parameters()):
                    swa_param.data.mul_(1.0 - virtual_decay).add_(virtual_decay, param.data)
                n_avg += 1

            if (step + 1) % eval_every == 0:
                if step / epoch_step > 3:
                    virtual_decay = 1 / float(n_avg + 1)
                    for swa_param, param in zip(swa_model.parameters(), model.parameters()):
                        swa_param.data.mul_(1.0 - virtual_decay).add_(virtual_decay, param.data)
                    n_avg += 1

                val_loss, score, post_score, other_score, align_score, union_score, post_diffs, other_diffs, align_diffs = validation(
                    model, val_loader, loss_fn, post_diffs, other_diffs, align_diffs, cv)

                if step / epoch_step >= 1.5:
                    swa_val_loss, swa_score, swa_post_score, swa_other_score, swa_align_score, swa_union_score, post_diffs, other_diffs, align_diffs = validation(
                        swa_model,
                        val_loader,
                        loss_fn,
                        post_diffs,
                        other_diffs,
                        align_diffs, cv)
                else:
                    swa_val_loss, swa_score, swa_post_score, swa_other_score, swa_align_score, swa_union_score = 0, 0, 0, 0, 0, 0

                if score > best_score:  # 记录未经过任何处理的best score
                    best_score = score
                    corr_post_score = post_score
                    corr_other_score = other_score
                    corr_align_score = align_score

                    if Save_ckpt:
                        model_path = "{}/{}_bert_fold_{}.pth".format(save_dir, cfg["output"], cv)
                        torch.save(model.state_dict(), model_path)

                if swa_score > swa_best_score:  # 记录未经过任何处理的best score
                    swa_best_score = swa_score
                    corr_swa_post_score = swa_post_score
                    corr_swa_other_score = swa_other_score
                    corr_swa_align_score = swa_align_score

                    if Save_ckpt:
                        swa_model_path = "{}/{}_swa_bert_fold_{}.pth".format(save_dir, cfg["output"], cv)
                        torch.save(swa_model.state_dict(), swa_model_path)

                elapsed_time = time.time() - start_time

                print(
                    '{:.1f}k  {:.1f}  |  {:.4f}  {:.4f}  {:.4f} {:.4f} {:.4f} {:.4f}|  {:.4f}  {:.4f}  {:.4f} {:.4f} {:.4f} {:.4f}| {:.4f}  | {:.1f}  | {:.7f}'.format(
                        step / 1000, step / epoch_step, val_loss, score, post_score, other_score, align_score,
                        union_score, swa_val_loss, swa_score, swa_post_score, swa_other_score, swa_align_score,
                        swa_union_score, sum_loss / (step + 1), elapsed_time, optimizer.param_groups[0]['lr']))

                model.train()
                swa_model.train()

        model.load_state_dict(torch.load("{}/{}_bert_fold_{}.pth".format(save_dir, cfg["output"], cv)))
        swa_model.load_state_dict(torch.load("{}/{}_swa_bert_fold_{}.pth".format(save_dir, cfg["output"], cv)))

        result, post_result, other_result, align_result = predict_result(model, test_loader)
        swa_result, swa_post_result, swa_other_result, swa_align_result = predict_result(swa_model, test_loader)

        if True:
            sub[target_columns] = result
            sub.to_csv("{}/fold_{}.csv".format(cfg["output"], cv), index=False)

            sub[target_columns] = post_result
            sub.to_csv("{}/post_fold_{}.csv".format(cfg["output"], cv), index=False)

            sub[target_columns] = other_result
            sub.to_csv("{}/other_fold_{}.csv".format(cfg["output"], cv), index=False)

            sub[target_columns] = align_result
            sub.to_csv("{}/align_fold_{}.csv".format(cfg["output"], cv), index=False)

            sub[target_columns] = swa_result
            sub.to_csv("{}/swa_fold_{}.csv".format(cfg["output"], cv), index=False)

            sub[target_columns] = swa_post_result
            sub.to_csv("{}/swa_post_fold_{}.csv".format(cfg["output"], cv), index=False)

            sub[target_columns] = swa_other_result
            sub.to_csv("{}/swa_other_fold_{}.csv".format(cfg["output"], cv), index=False)

            sub[target_columns] = swa_align_result
            sub.to_csv("{}/swa_align_fold_{}.csv".format(cfg["output"], cv), index=False)

        results += result
        post_results += post_result
        other_results += other_result
        align_results += align_result

        swa_results += swa_result
        swa_post_results += swa_post_result
        swa_other_results += swa_other_result
        swa_align_results += swa_align_result

        average_spearman += best_score
        average_post_score += corr_post_score
        average_other_score += corr_other_score
        average_align_score += corr_align_score
        average_swa_score += swa_best_score
        average_swa_post_score += corr_swa_post_score
        average_swa_other_score += corr_swa_other_score
        average_swa_align_score += corr_swa_align_score

        torch.cuda.empty_cache()

    print("*********************average******************")
    average_spearman /= NFOLDS
    average_post_score /= NFOLDS
    average_other_score /= NFOLDS
    average_align_score /= NFOLDS
    average_swa_score /= NFOLDS
    average_swa_post_score /= NFOLDS
    average_swa_other_score /= NFOLDS
    average_swa_align_score /= NFOLDS
    print(
        "{} {} {} {} {} {} {} {}".format(average_spearman, average_post_score, average_other_score, average_align_score,
                                         average_swa_score, average_swa_post_score, average_swa_other_score,
                                         average_swa_align_score))

    return results / NFOLDS, post_results / NFOLDS, other_results / NFOLDS, align_results / NFOLDS, swa_results / NFOLDS, swa_post_results / NFOLDS, swa_other_results / NFOLDS, swa_align_results / NFOLDS


x_test = convert_lines(tokenizer, test['question_title'], test['question_body'], test['answer'])
test_loader = torch.utils.data.DataLoader(QuestDataset(x_test),
                                          batch_size=BATCH_SIZE, shuffle=False)

if True:
    preds, post_preds, other_preds, align_preds, swa_preds, swa_post_preds, swa_other_preds, swa_align_preds = train_model(
        tokenizer, train, save_dir, test_loader)

    sub[target_columns] = preds
    sub.to_csv("{}/result.csv".format(cfg["output"]), index=False)

    sub[target_columns] = post_preds
    sub.to_csv("{}/post.csv".format(cfg["output"]), index=False)

    sub[target_columns] = other_preds
    sub.to_csv("{}/other.csv".format(cfg["output"]), index=False)

    sub[target_columns] = align_preds
    sub.to_csv("{}/align.csv".format(cfg["output"]), index=False)

    sub[target_columns] = swa_preds
    sub.to_csv("{}/swa_result.csv".format(cfg["output"]), index=False)

    sub[target_columns] = swa_post_preds
    sub.to_csv("{}/swa_post.csv".format(cfg["output"]), index=False)

    sub[target_columns] = swa_other_preds
    sub.to_csv("{}/swa_other.csv".format(cfg["output"]), index=False)

    sub[target_columns] = swa_align_preds
    sub.to_csv("{}/swa_align.csv".format(cfg["output"]), index=False)

if cfg["Use_semi_supervised"]:
    save_dir += "_semi"

    print("read scv from {}".format(cfg["output"]))
    test_label = pd.read_csv(cfg["output"])

    print(test_label.columns)
    print(test_label.shape)
    print(test_label.head)

    test = pd.merge(test, test_label, how='left', on='qa_id')
    print(test.head)

    frames = [train, test]

    train = pd.concat(frames)

    preds = train_model(tokenizer, train, save_dir, test_loader)


