import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from stratifiers import MultilabelStratifiedKFold
import random
import numpy as np
import tensorflow as tf
import bert_tokenization as tokenization
import tensorflow.keras.backend as K
import gc
import os
from scipy.stats import spearmanr
from bert_utils import compute_1sen_input_arrays, compute_output_arrays
from math import floor, ceil

np.set_printoptions(suppress=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
SAVE_MODELS = True
learning_rate = 5e-5
epochs = 3
batch_size = 8
num_folds = 3
actual_folds = 5
seed = 2019

PATH = '../input/google-quest-challenge/'
BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
df_sub = pd.read_csv('online-sub.csv')
tokenizer = tokenization.FullTokenizer(BERT_PATH + '/assets/vocab.txt', True)
max_seq_len = 512

output_categories = ['question_type_spelling']
input_categories = ['question_title', 'question_body']

print(df_sub.head())
print('\noutput categories:\n\t', output_categories)
print('\ninput categories:\n\t', input_categories)
df_test = pd.read_csv('../input/google-quest-challenge/test.csv')

test_inputs = compute_1sen_input_arrays(df_test, input_categories, tokenizer, max_seq_len)


def bert_model():
    input_word_ids = tf.keras.Input(
        (max_seq_len,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.Input(
        (max_seq_len,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.Input(
        (max_seq_len,), dtype=tf.int32, name='input_segments')

    bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)

    _, sequence_output = bert_layer([input_word_ids, input_masks, input_segments])

    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="dense_output")(x)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_masks, input_segments], outputs=out)

    return model


folds = [0, 1, 2]

test_predictions = []
# for fold, epoch in ms:
for fold in folds:
    model = bert_model()
    # 加载模型参数
    p = f'spelling-models/fold-{fold}-best.h5'
    print(f"Loading model from: {p}")
    model.load_weights(p)
    print("Predicting ......")
    tmp_test_inputs = [np.array(arr) for arr in test_inputs]
    pred = model.predict(tmp_test_inputs, batch_size=batch_size)
    test_predictions.append(np.array(pred))
    del p, tmp_test_inputs, pred, model
    K.clear_session()
    [gc.collect() for _ in range(15)]

pred = np.average(test_predictions, axis=0)
assert df_sub.iloc[:, 20].name == 'question_type_spelling'
df_sub.iloc[:, 20] = pred
df_sub.to_csv('submission.csv', index=False)
print("Done!")
