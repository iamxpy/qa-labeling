import argparse
from collections import defaultdict, Counter
import random

import pandas as pd
from tqdm import tqdm
# from common import *
# from dataset import *
import os


def make_folds(n_folds: int) -> pd.DataFrame:
    all_q = [[], [], [], [], []]
    df = pd.read_csv('./google-quest-challenge/train.csv')
    cls_counts = {}
    for key in targets:
        cls_counts[key] = 0

    for key in targets:
        cls_counts[key] = int(df[df[key] > 0].shape[0])

    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in tqdm(df.sample(frac=1, random_state=42).itertuples(),
                     total=len(df)):
        tem = []
        for i in range(len(targets)):
            if item[12 + i] > 0:
                tem.append(targets[i])

        cls = min(tem, key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        flag = 0

        for i in range(n_folds):
            if item.question_title in all_q[i]:
                fold = i
                flag = 1
                break

        if flag == 0:
            fold = random.choice([f for f, count in fold_counts
                                  if count == min_count])

        all_q[fold].append(item.question_title)
        folds[item.Index] = fold
        for cls in tem:
            fold_cls_counts[fold, cls] += 1

    df['fold'] = folds
    print(len(df))
    return df

targets = [
    'question_asker_intent_understanding',
    'question_body_critical',
    'question_conversational',
    'question_expect_short_answer',
    'question_fact_seeking',
    'question_has_commonly_accepted_answer',
    'question_interestingness_others',
    'question_interestingness_self',
    'question_multi_intent',
    'question_not_really_a_question',
    'question_opinion_seeking',
    'question_type_choice',
    'question_type_compare',
    'question_type_consequence',
    'question_type_definition',
    'question_type_entity',
    'question_type_instructions',
    'question_type_procedure',
    'question_type_reason_explanation',
    'question_type_spelling',
    'question_well_written',
    'answer_helpful',
    'answer_level_of_information',
    'answer_plausible',
    'answer_relevance',
    'answer_satisfaction',
    'answer_type_instructions',
    'answer_type_procedure',
    'answer_type_reason_explanation',
    'answer_well_written'
]


def main():
    path = os.getcwd()  # 获取当前路径
    print(path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()
    df = make_folds(n_folds=args.n_folds)
    df.to_csv('./google-quest-challenge/folds_trans.csv', index=None)


if __name__ == '__main__':
    main()
