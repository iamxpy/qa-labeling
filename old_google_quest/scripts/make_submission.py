import os
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import combinations

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

folds = 10

ss = [pd.read_csv(f'submission-fold-{fold}.csv') for fold in range(folds)]

scores = pd.read_csv('models/scores.csv')

mats = [pd.concat([ss[i].iloc[:, j] for i in range(folds)], axis=1) for j in range(1, 31)]

for i in range(30):
    mats[i].columns = [f'fold-{fold}' for fold in range(folds)]

corrs = [mats[i].corr() for i in range(30)]

min_corr_folds = list(range(30))
print("Calculating ......")
for i, corr in enumerate(corrs):
    min_corr = float('inf')
    mcf = None
    for idx in combinations(range(folds), 5):
        c = corr.iloc[list(idx), list(idx)]
        corr_sum = np.sum(c.sum())
        if corr_sum < min_corr:
            min_corr = corr_sum
            min_corr_folds[i] = list(idx)

min_corr_folds = pd.DataFrame(min_corr_folds, columns=list('abcde'))

avg_preds = ss[0].copy()

for c in range(30):
    cols = [ss[i].iloc[:, c + 1] for i in min_corr_folds.iloc[c, :]]
    scs = scores.iloc[0, min_corr_folds.iloc[c, :]]
    weights = tf.nn.softmax(scs).numpy()
    col = np.average(cols, axis=0, weights=weights)
    avg_preds.iloc[:, c + 1] = col

avg_preds.to_csv('submission.csv', index=False)
print('Done.')
