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
from math import floor, ceil

np.set_printoptions(suppress=True)

SAVE_MODELS = True
learning_rate = 5e-5
epochs = 10  # 考虑保持或继续增大
batch_size = 8
num_folds = 5  # 考虑适当增大
actual_folds = 5
seed = 2019

PATH = '../input/google-quest-challenge/'
BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
tokenizer = tokenization.FullTokenizer(BERT_PATH + '/assets/vocab.txt', True)
MAX_SEQUENCE_LENGTH = 512


df_train = pd.read_csv(PATH + 'train.csv').head(20)
df_sub = pd.read_csv(PATH + 'sample_submission.csv')
print('train shape =', df_train.shape)

output_categories = list(df_train.columns[11:])
input_categories = list(df_train.columns[[1, 2, 5]])
print('\noutput categories:\n\t', output_categories)
print('\ninput categories:\n\t', input_categories)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(seed)

def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))  # [PAD]对应segment id为0


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))  # [PAD]对应vocab id为0
    return input_ids


def _tokenize_trim(title, question, answer, max_sequence_length,
                   t_max_len=30, q_max_len=239, a_max_len=239):
    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)

    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len + q_len + a_len + 4) > max_sequence_length:

        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len) / 2)
            q_max_len = q_max_len + ceil((t_max_len - t_len) / 2)
        else:
            t_new_len = t_max_len

        if a_max_len > a_len:
            a_new_len = a_len
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len

        if t_new_len + a_new_len + q_new_len + 4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d"
                             % (max_sequence_length, (t_new_len + a_new_len + q_new_len + 4)))

        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]

    return t, q, a


def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, row in df[columns].iterrows():
        t, q, a = row.question_title, row.question_body, row.answer

        t, q, a = _tokenize_trim(t, q, a, max_sequence_length)

        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32),
            np.asarray(input_segments, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, batch_size=16, fold=None):
        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]

        self.batch_size = batch_size
        self.fold = fold
        self.val_loss = float('inf')
        self.rho_value = -1
        self.valid_predictions = []
        # self.test_predictions = 0
        self.best_valid_predictions = None
        self.train_rho_values = []

    def on_epoch_end(self, epoch, logs={}):
        val_pred = self.model.predict(self.valid_inputs, batch_size=self.batch_size)
        self.valid_predictions.append(val_pred)
        rho_val = compute_spearmanr(
            # self.valid_outputs, np.average(self.valid_predictions, axis=0))
            self.valid_outputs, val_pred)
        self.train_rho_values.append(rho_val)
        # check whether to save model and update best rho value
        if rho_val >= self.rho_value:
            self.rho_value = rho_val
            self.best_valid_predictions = val_pred
            if self.fold is not None:
                if not os.path.exists('models/'):
                    os.mkdir('models/')
                self.model.save_weights(f'models/bert-base-fold-{fold}.h5')

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
    out = tf.keras.layers.Dense(30, activation="sigmoid", name="dense_output")(x)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_masks, input_segments], outputs=out)

    return model


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


# #### 4. Obtain inputs and targets, as well as the indices of the train/validation splits

# In[5]:


outputs = compute_output_arrays(df_train, output_categories)
inputs = compute_input_arays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

# kf = KFold(n_splits=5, random_state=2019, shuffle=True).split(X=df_train.question_body)
# kf = GroupKFold(n_splits=10).split(X=df_train.question_body, groups=df_train.question_body)
kf = MultilabelStratifiedKFold(n_splits=num_folds).split(df_train, outputs)

# #### 5. Training, validation and testing
# 
# Loops over the folds in gkf and trains each fold for 5 epochs --- with a learning rate of 1e-5 and batch_size of 8. A simple binary crossentropy is used as the objective-/loss-function. 

# In[6]:

histories = []
cv_scores = []
pred_train = np.zeros([df_train.shape[0], len(output_categories)])
train_rho_values = []
test_predictions = []

# for fold, (train_idx, valid_idx) in enumerate(gkf):
for fold, (train_idx, valid_idx) in enumerate(kf):

    if fold >= actual_folds:
        break

    model = bert_model()

    train_inputs = [inputs[i][train_idx] for i in range(3)]
    train_outputs = outputs[train_idx]

    valid_inputs = [inputs[i][valid_idx] for i in range(3)]
    valid_outputs = outputs[valid_idx]

    # history contains two lists of valid and test preds respectively:
    #  [valid_predictions_{fold}, test_predictions_{fold}]
    if not SAVE_MODELS:
        fold = None
    history = train_and_predict(model,
                                train_data=(train_inputs, train_outputs),
                                valid_data=(valid_inputs, valid_outputs),
                                # learning_rate=3e-5, epochs=3, batch_size=8, patience=5,
                                # learning_rate=1e-5, epochs=4, batch_size=8, patience=float('inf'),
                                learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                                loss_function='binary_crossentropy', fold=fold)
    pred_train[valid_idx, :] = np.array(history.best_valid_predictions)  # return a copy, so we can delete 'history'
    test_predictions.append(np.array(history.test_predictions))  # use a copy, so we can delete 'history'
    # pred_test += np.array(history.test_predictions)
    train_rho_values.append(np.array(history.train_rho_values))
    cv_scores.append(history.rho_value)  # rho_value is only a scalar, not an object, so need not to copy it

    # collect memory
    del train_inputs, train_outputs, valid_inputs, valid_outputs, history, model
    K.clear_session()
    [gc.collect() for _ in range(15)]

# #### 6. Process and submit test predictions
# 
# First the test predictions are read from the list of lists of `histories`. Then each test prediction list (in lists) is averaged. Then a mean of the averages is computed to get a single prediction for each data point. Finally, this is saved to `submission.csv`

# In[7]:


print(f"cv scores: {cv_scores}")

if not SAVE_MODELS:
    ps = tf.nn.softmax(train_rho_values).numpy()
    test_predictions = [np.average(test_predictions[i], axis=0, weights=ps[i]) for i in range(len(test_predictions))]
    test_predictions = np.mean(test_predictions, axis=0)
    df_sub.iloc[:, 1:] = test_predictions

    df_sub.to_csv('submission.csv', index=False)
else:
    pred_train_df = pd.DataFrame(pred_train, columns=output_categories)

    pred_train_df.to_csv('pred_train.csv', index=False)

# In[ ]:


# In[ ]:
