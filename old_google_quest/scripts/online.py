import pandas as pd
import numpy as np

pred_df = pd.read_csv('../input/csv-files/submission.csv')
pred = pred_df.values
probs = pred[:, 1:]


def fix(candidate, expected_values):
    gaps = np.abs(expected_values - candidate)
    fixed = expected_values[np.argmin(gaps)]
    return fixed


values = []
denominator = [6, 9, 10, 15]
for f in denominator:
    for i in range(f):
        a = i / f
        if a not in values:
            values.append(a)
values = np.array(values)

for i in range(len(pred)):
    for j in range(0, 30):
        probs[i][j] = fix(probs[i][j], values)

probs += 1e-7 * np.abs(np.random.normal(size=probs.shape))
probs[probs <= 0] = 1e-5
probs[probs >= 1] = 1 - 1e-5

pred[:, 1:] = probs

sample = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
n = 0
for line in pred:
    for i in range(len(sample)):
        if (sample.loc[i]['qa_id'] == int(line[0])):
            for j in range(1, 31):
                sample.iloc[i, j] = line[j]
            break
print(sample.head())
sample.to_csv('submission.csv', index=False)
print("Done!")
