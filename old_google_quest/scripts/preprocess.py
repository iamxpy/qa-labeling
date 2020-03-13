# coding=utf-8

import os
import pandas as pd
from stratifiers import MultilabelStratifiedKFold
from bert_utils import compute_output_arrays

num_folds = 10
seed = 2019
PATH = '../input/google-quest-challenge/'

df_train = pd.read_csv(PATH + 'train.csv')
df_test = pd.read_csv(PATH + 'test.csv')
outputs = compute_output_arrays(df_train, list(df_train.columns[11:]))

kf = MultilabelStratifiedKFold(n_splits=num_folds).split(df_train, outputs)

for fold, (train_idx, valid_idx) in enumerate(kf):
    print(f"Processing fold {fold}")
    saved_dir = PATH + f'fold-{fold}/'
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    sub_df_train = df_train.iloc[train_idx, :]
    sub_df_dev = df_train.iloc[valid_idx, :]
    sub_df_train.to_csv(saved_dir + 'train.csv', index=False)
    sub_df_dev.to_csv(saved_dir + 'dev.csv', index=False)
    df_test.to_csv(saved_dir + 'test.csv', index=False)

print("Done!")
