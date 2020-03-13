#!/usr/bin/env python
# coding: utf-8

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
tokenizer = tokenization.FullTokenizer(BERT_PATH + '/assets/vocab.txt', True)
MAX_SEQUENCE_LENGTH = 512

df_train = pd.read_csv(PATH + 'spelling_train.csv')
df_sub = pd.read_csv(PATH + 'sample_submission.csv')
print('train shape =', df_train.shape)

output_categories = ['question_type_spelling']
input_categories = ['question_title', 'question_body']
print('\noutput categories:\n\t', output_categories)
print('\ninput categories:\n\t', input_categories)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(seed)


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, batch_size=16, eval_every=10, fold=None):
        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]

        self.batch_size = batch_size
        self.fold = fold
        self.val_loss = float('inf')
        self.rho_value = -1
        self.eval_every = eval_every
        self.valid_predictions = []
        self.train_rho_values = []

    def on_batch_end(self, batch, logs={}):
        if batch > 0 and batch % self.eval_every == 0:
            val_pred = self.model.predict(self.valid_inputs, batch_size=self.batch_size)
            self.valid_predictions.append(val_pred)
            rho_val = compute_spearmanr(
                # self.valid_outputs, np.average(self.valid_predictions, axis=0))
                self.valid_outputs, val_pred)
            self.train_rho_values.append(rho_val)
            # check whether to save model and update best rho value
            if rho_val >= self.rho_value:
                self.rho_value = rho_val
                if self.fold is not None:
                    if not os.path.exists('spelling-models/'):
                        os.mkdir('spelling-models/')
                    self.model.save_weights(f'spelling-models/fold-{fold}-best.h5')

            print("\nvalidation rho: %.4f" % rho_val)


def bert_model():
    input_word_ids = tf.keras.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')

    # TODO DEBUG in CPU
    # bert_layer = fake_bert_for_debug()
    bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)

    _, sequence_output = bert_layer([input_word_ids, input_masks, input_segments])

    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="dense_output")(x)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_masks, input_segments], outputs=out)

    return model


aug_columns = [[f'question_title_{lan}', f'question_body_{lan}'] for lan in ['ar', 'de', 'es', 'fr', 'jp', 'ru']]


def aug_data(df):
    res = df.copy()
    for i, row in df.iterrows():
        if isinstance(row.question_title_fr, str):
            for cs in aug_columns:
                aug = pd.DataFrame(row[cs + ['question_type_spelling']]).T
                aug.columns = ['question_title', 'question_body', 'question_type_spelling']
                res = res.append(aug, sort=True)
    return res


def train_and_predict(model, train_data, valid_data,
                      learning_rate, epochs, batch_size, loss_function, fold):
    custom_callback = CustomCallback(
        valid_data=(valid_data[0], valid_data[1]),
        batch_size=batch_size,
        fold=fold)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer)
    model.fit(train_data[0], train_data[1], epochs=epochs, verbose=1,
              batch_size=batch_size, callbacks=[custom_callback])

    return custom_callback


# outputs = compute_output_arrays(df_train, output_categories)
# inputs = compute_1sen_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

# kf = KFold(n_splits=5, random_state=2019, shuffle=True).split(X=df_train.question_body)
kf = GroupKFold(n_splits=num_folds).split(X=df_train.question_body, groups=df_train.question_body)
# kf = MultilabelStratifiedKFold(n_splits=num_folds).split(df_train, outputs)

histories = []
cv_scores = []
train_rho_values = []

# for fold, (train_idx, valid_idx) in enumerate(gkf):
for fold, (train_idx, valid_idx) in enumerate(kf):

    if fold >= actual_folds:
        break

    model = bert_model()

    train_set = df_train.iloc[train_idx]
    train_set = aug_data(train_set)
    train_set.reset_index(drop=True, inplace=True)
    train_inputs = compute_1sen_input_arrays(train_set, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    train_outputs = compute_output_arrays(train_set, output_categories)

    valid_set = df_train.iloc[valid_idx]
    valid_inputs = compute_1sen_input_arrays(valid_set, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    valid_outputs = compute_output_arrays(valid_set, output_categories)

    # history contains two lists of valid and test preds respectively:
    if not SAVE_MODELS:
        fold = None
    history = train_and_predict(model,
                                train_data=(train_inputs, train_outputs),
                                valid_data=(valid_inputs, valid_outputs),
                                # learning_rate=3e-5, epochs=3, batch_size=8, patience=5,
                                # learning_rate=1e-5, epochs=4, batch_size=8, patience=float('inf'),
                                learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                                loss_function='binary_crossentropy', fold=fold)
    train_rho_values.append(np.array(history.train_rho_values))
    cv_scores.append(history.rho_value)  # rho_value is only a scalar, not an object, so need not to copy it

    # collect memory
    del train_inputs, train_outputs, valid_inputs, valid_outputs, history, model
    K.clear_session()
    [gc.collect() for _ in range(15)]

print(f"cv scores: {cv_scores}")
#
# if not SAVE_MODELS:
#     ps = tf.nn.softmax(train_rho_values).numpy()
#     test_predictions = [np.average(test_predictions[i], axis=0, weights=ps[i]) for i in range(len(test_predictions))]
#     test_predictions = np.mean(test_predictions, axis=0)
#     df_sub.iloc[:, 1:] = test_predictions
#
#     df_sub.to_csv('submission.csv', index=False)
# else:
#     pred_train_df = pd.DataFrame(pred_train, columns=output_categories)
#
#     pred_train_df.to_csv('pred_train.csv', index=False)

# In[ ]:


# In[ ]:
