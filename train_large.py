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

save_dir = "./ck_fold_{}_epoch_{}_LR_{}_layer_{}".format(NFOLDS, cfg["EPOCH"], LR, Last_Layer)

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
target_columns = sub.columns.values[1:].tolist()

train = pd.read_csv(f'{DATA_DIR}/folds_trans_more.csv')
train['train'] = True
test = pd.read_csv(f'{DATA_DIR}/test.csv')

use_small_train = True
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

            test_preds[idx * INFER_BATHC_SIZE: min((idx + 1) * INFER_BATHC_SIZE, test_len)] = pred.detach().cpu().numpy()

    test_preds_new = test_preds.copy()
    align_preds = test_preds.copy()

    post_test_preds = deal_result(test_preds, ban_list)
    other_test_preds = post_deal_result(test_preds_new, ban_list)
    align_test_preds = align(align_preds, ban_list)
    return test_preds, post_test_preds, other_test_preds, align_test_preds


def cal(arr1, arr2):
    return np.array([compute_spearmanr(arr1[:, i].reshape(-1, 1), arr2[:, i].reshape(-1, 1)) for i in range(30)])


def validation(model, val_loader, loss_fn, ori_scores=None, post_scores=None, other_scores=None, align_scores=None,
               fold=None):
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
    if fold is not None:
        ori_scores_row = cal(true_label, valid_preds)
        row = pd.DataFrame(ori_scores_row).T
        row.columns = output_categories
        ori_scores = ori_scores.append(row)
        ori_scores.to_csv("{}/fold_{}_ori_scores.csv".format(cfg["output"], fold), index=False)

    # post without ban socres
    valid_preds_post = deal_result(valid_preds_post)
    if fold is not None:
        post_scores_row = cal(true_label, valid_preds_post)
        row = pd.DataFrame(post_scores_row).T
        row.columns = output_categories
        post_scores = post_scores.append(row)
        post_scores.to_csv("{}/fold_{}_post_scores.csv".format(cfg["output"], fold), index=False)
    # post with ban
    for c in ban_list:
        valid_preds_post[:, c] = valid_preds[:, c]
    post_score = compute_spearmanr(true_label, valid_preds_post)

    # other without ban scores
    valid_preds_other = post_deal_result(valid_preds_other)
    if fold is not None:
        other_scores_row = cal(true_label, valid_preds_other)
        row = pd.DataFrame(other_scores_row).T
        row.columns = output_categories
        other_scores = other_scores.append(row)
        other_scores.to_csv("{}/fold_{}_other_scores.csv".format(cfg["output"], fold), index=False)
    # other with ban
    for c in ban_list:
        valid_preds_other[:, c] = valid_preds[:, c]
    other_score = compute_spearmanr(true_label, valid_preds_other)
    # union form other
    for c in union_from_other:
        valid_preds_union[:, c] = valid_preds_other[:, c]

    # align without ban scores
    valid_preds_align = align(valid_preds_align)
    if fold is not None:
        align_scores_row = cal(true_label, valid_preds_align)
        row = pd.DataFrame(align_scores_row).T
        row.columns = output_categories
        align_scores = align_scores.append(row)
        align_scores.to_csv("{}/fold_{}_align_scores.csv".format(cfg["output"], fold), index=False)
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
        val_loader), score, post_score, other_score, align_score, union_score, ori_scores, post_scores, other_scores, align_scores


def optim_parm_collect(model, LR, pre_model):
    if torch.cuda.device_count() > 1:
        if "albert" in cfg["pretrained_model"]:
            optimizer_grouped_parameters = model.parameters()
        elif "bert" in cfg["pretrained_model"]:
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
        if "albert" in cfg["pretrained_model"]:
            optimizer_grouped_parameters = model.parameters()
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
    average_post_score = 0
    average_other_score = 0
    average_align_score = 0

    results = np.zeros((len(test), len(target_columns)))
    other_results = np.zeros((len(test), len(target_columns)))
    post_results = np.zeros((len(test), len(target_columns)))
    align_results = np.zeros((len(test), len(target_columns)))


    for cv in range(NFOLDS):
        print(f'fold {cv + 1}')

        ori_scores = pd.DataFrame(columns=output_categories)
        post_scores = pd.DataFrame(columns=output_categories)
        other_scores = pd.DataFrame(columns=output_categories)
        align_scores = pd.DataFrame(columns=output_categories)

        train_df = train[train['fold'] != cv]
        train_df_more = train_df.copy(deep=True)

        if cfg["upsample"] != None:
            for i, row in train_df.iterrows():
                if row['train']:
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

        val_loader = DataLoader(QuestDataset(x_eval, y_eval), batch_size=INFER_BATHC_SIZE, shuffle=False, pin_memory=True)

        STEPS = cfg["EPOCH"] * math.ceil(len(train_loader))

        print("tot steps {}".format(STEPS))
        model = BertForQuest()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        print(
            'iter epoch | valid_loss  spearmanr post_score other_score align_score union_score | train_loss |  time          ')
        print(
            '------------------------------------------------------------------------------------------------------------------------------------------------')

        model = model.to(device)

        loss_fn = torch.nn.BCELoss()

        LR = 3e-5
        optimizer_grouped_parameters = optim_parm_collect(model, LR, pre_model)
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=LR)

        eval_every = 800
        best_score = 0.0
        corr_post_score = 0.0
        corr_other_score = 0.0
        corr_align_score = 0.0

        sum_loss = 0.0

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
            if (step + 1) % BACK_STEP == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (step+1) % eval_every == 0:
                val_loss, score, post_score, other_score, align_score, union_score, ori_scores, post_scores, other_scores, align_scores = validation(
                    model, val_loader, loss_fn, ori_scores, post_scores, other_scores, align_scores, cv)

                if score > best_score:  # 记录未经过任何处理的best score
                    best_score = score
                    corr_post_score = post_score
                    corr_other_score = other_score
                    corr_align_score = align_score

                    if Save_ckpt:
                        model_path = "{}/{}_bert_fold_{}.pth".format(save_dir, cfg["output"], cv)
                        torch.save(model.state_dict(), model_path)

                elapsed_time = time.time() - start_time

                print(
                    '{:.1f}k  {:.1f}  |  {:.4f}  {:.4f}  {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f}  | {:.1f}  | {:.7f}'.format(
                        step / 1000, step / epoch_step, val_loss, score, post_score, other_score, align_score,
                        union_score, sum_loss / (step + 1), elapsed_time, optimizer.param_groups[0]['lr']))

                model.train()

        model.load_state_dict(torch.load("{}/{}_bert_fold_{}.pth".format(save_dir, cfg["output"], cv)))

        result, post_result, other_result, align_result = predict_result(model, test_loader)

        if True:
            sub[target_columns] = result
            sub.to_csv("{}/fold_{}.csv".format(cfg["output"], cv), index=False)

            sub[target_columns] = post_result
            sub.to_csv("{}/post_fold_{}.csv".format(cfg["output"], cv), index=False)

            sub[target_columns] = other_result
            sub.to_csv("{}/other_fold_{}.csv".format(cfg["output"], cv), index=False)

            sub[target_columns] = align_result
            sub.to_csv("{}/align_fold_{}.csv".format(cfg["output"], cv), index=False)

        results += result
        post_results += post_result
        other_results += other_result
        align_results += align_result

        average_spearman += best_score
        average_post_score += corr_post_score
        average_other_score += corr_other_score
        average_align_score += corr_align_score

        torch.cuda.empty_cache()

    print("*********************average******************")
    average_spearman /= NFOLDS
    average_post_score /= NFOLDS
    average_other_score /= NFOLDS
    average_align_score /= NFOLDS
    print(
        "{} {} {} {}".format(average_spearman, average_post_score, average_other_score, average_align_score))

    return results / NFOLDS, post_results / NFOLDS, other_results / NFOLDS, align_results / NFOLDS


x_test = convert_lines(tokenizer, test['question_title'], test['question_body'], test['answer'])
test_loader = torch.utils.data.DataLoader(QuestDataset(x_test),
                                          batch_size=INFER_BATHC_SIZE, shuffle=False)

if False:
    preds, post_preds, other_preds, align_preds = train_model(
        tokenizer, train, save_dir, test_loader)

    sub[target_columns] = preds
    sub.to_csv("{}/result.csv".format(cfg["output"]), index=False)

    sub[target_columns] = post_preds
    sub.to_csv("{}/post.csv".format(cfg["output"]), index=False)

    sub[target_columns] = other_preds
    sub.to_csv("{}/other.csv".format(cfg["output"]), index=False)

    sub[target_columns] = align_preds
    sub.to_csv("{}/align.csv".format(cfg["output"]), index=False)


if cfg["Use_semi_supervised"]:
    save_dir += "_semi"

    test_label = pd.read_csv("{}/align.csv".format(cfg["output"]))

    train = pd.read_csv(f'{DATA_DIR}/train.csv')

    train = make_dataset(train, test, test_label)

    cfg["output"] += "_semi"
    preds, post_preds, other_preds, align_preds = train_model(
        tokenizer, train, save_dir, test_loader)

    sub[target_columns] = preds
    sub.to_csv("{}/result.csv".format(cfg["output"]), index=False)

    sub[target_columns] = post_preds
    sub.to_csv("{}/post.csv".format(cfg["output"]), index=False)

    sub[target_columns] = other_preds
    sub.to_csv("{}/other.csv".format(cfg["output"]), index=False)

    sub[target_columns] = align_preds
    sub.to_csv("{}/align.csv".format(cfg["output"]), index=False)