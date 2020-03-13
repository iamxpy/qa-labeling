#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import tensorflow_hub as hub
from stratifiers import MultilabelStratifiedKFold
import random
import numpy as np
import tensorflow as tf
import bert_tokenization as tokenization
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import gc
import os
from scipy.stats import spearmanr
from math import floor, ceil
from bert_utils import compute_output_arrays, compute_split_input_arrays, get_recommended_batch_size_with_12GB
from accum_optimizer import AccumOptimizer

np.set_printoptions(suppress=True)

SAVE_MODELS = False
LONG_SENTENCE_MAX_LEN = 1600
num_split = 4
expected_least_batch_size = 4
assert LONG_SENTENCE_MAX_LEN % num_split == 0
average_short_sentence_len = 60
BERT_SEQUENCE_LEN = (LONG_SENTENCE_MAX_LEN // num_split) + average_short_sentence_len
assert BERT_SEQUENCE_LEN <= 512
learning_rate = 3e-5
equal_epochs = 5
batch_size = get_recommended_batch_size_with_12GB(BERT_SEQUENCE_LEN) // (2 * num_split)
print(f"\nUsing actual batch_size: {batch_size}")
accumulation_step = int(ceil(expected_least_batch_size / batch_size))
print(f"Expected least batch_size: {expected_least_batch_size}")
print(f"Using Accumulation batch_size: {accumulation_step * batch_size}")
epochs = accumulation_step * equal_epochs
print(f"Using Accumulation epochs {epochs}, equals to {equal_epochs} without accumulation.")
num_folds = 10
start_fold = 0
actual_fold_num = 1
seed = 2019

PATH = '../input/google-quest-challenge/'
BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
tokenizer = tokenization.FullTokenizer(BERT_PATH + '/assets/vocab.txt', True)

df_train = pd.read_csv(PATH + 'train.csv')
df_test = pd.read_csv(PATH + 'test.csv')
df_sub = pd.read_csv(PATH + 'sample_submission.csv')
print('train shape =', df_train.shape)
print('test shape =', df_test.shape)

output_categories = list(df_train.columns[11:])

question_input_categories = list(df_train.columns[[1, 2]])
qa_input_categories = list(df_train.columns[[1, 5]])
print('\noutput categories:\n\t', output_categories)

question_categories = list(df_train.columns[11:32])  # 21 categories
qa_categories = list(df_train.columns[32:])  # 9 categories


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def restrict_tf_memory_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


seed_everything(seed)
restrict_tf_memory_usage()


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, test_data, batch_size=16, fold=None):
        super().__init__()
        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.test_inputs = test_data
        self.batch_size = batch_size
        self.fold = fold
        self.val_loss = float('inf')
        self.rho_value = -1
        self.valid_predictions = []
        self.test_predictions = []
        self.best_valid_predictions = None
        self.train_rho_values = []

    def on_epoch_end(self, epoch, logs={}):
        val_pred = self.model.predict(self.valid_inputs, batch_size=self.batch_size)
        self.valid_predictions.append(val_pred)
        rho_val = compute_spearmanr(
            self.valid_outputs, np.average(self.valid_predictions, axis=0))
        self.train_rho_values.append(rho_val)
        self.test_predictions.append(self.model.predict(self.test_inputs, batch_size=self.batch_size))
        if self.fold is not None:
            if not os.path.exists('models/'):
                os.mkdir('models/')
            self.model.save_weights(f'models/fold-{fold}-epoch-{epoch}.h5')  # 一共保存fold * epoch个模型
        if rho_val >= self.rho_value:
            self.rho_value = rho_val
            self.best_valid_predictions = val_pred
        print("\nvalidation rho: %.4f" % rho_val)


def multi_split_bert_model():
    # ids, masks, segments
    # shape: [batch_size, num_split, BERT_SEQ_LEN]
    question_ids = tf.keras.Input((num_split, BERT_SEQUENCE_LEN), dtype='int32', name='question_ids')
    question_masks = tf.keras.Input((num_split, BERT_SEQUENCE_LEN), dtype='int32', name='question_masks')
    question_segments = tf.keras.Input((num_split, BERT_SEQUENCE_LEN), dtype='int32', name='question_segments')

    qa_ids = tf.keras.Input((num_split, BERT_SEQUENCE_LEN), dtype='int32', name='qa_id')
    qa_masks = tf.keras.Input((num_split, BERT_SEQUENCE_LEN), dtype='int32', name='qa_mask')
    qa_segments = tf.keras.Input((num_split, BERT_SEQUENCE_LEN), dtype='int32', name='qa_segments')

    # Concat: 2 x [batch_size, num_split, BERT_SEQ_LEN] -> [2*batch_size, num_split, BERT_SEQ_LEN]
    # Reshape: [2*batch_size, num_split, BERT_SEQ_LEN] -> [2*batch_size*num_split, BERT_SEQ_LEN]
    bert_ids = tf.reshape(layers.Concatenate(axis=0)([question_ids, qa_ids]), [-1, BERT_SEQUENCE_LEN])
    # bert_ids = layers.Reshape([-1, BERT_SEQUENCE_LEN])(layers.Concatenate(axis=0)([question_ids, qa_ids]))
    bert_masks = tf.reshape(layers.Concatenate(axis=0)([question_masks, qa_masks]), [-1, BERT_SEQUENCE_LEN])
    # bert_masks = layers.Reshape([-1, BERT_SEQUENCE_LEN])(layers.Concatenate(axis=0)([question_masks, qa_masks]))
    bert_segments = tf.reshape(layers.Concatenate(axis=0)([question_segments, qa_segments]), [-1, BERT_SEQUENCE_LEN])
    # bert_segments = layers.Reshape([-1, BERT_SEQUENCE_LEN])(
    #     layers.Concatenate(axis=0)([question_segments, qa_segments]))

    # 最大程度的共享参数：对(title+body)和(title+answer)两个部分共用；对每个部分的所有split之间共用
    # todo 考虑对只共享每个部分内部的所有split，(title+body)和(title+answer)两个部分分别使用单独的bert
    bert = hub.KerasLayer(BERT_PATH, trainable=True)

    # [2*batch_size*num_split, BERT_SEQ_LEN] -> [2*batch_size*num_split, BERT_SEQ_LEN, 768]
    _, sequence_output = bert([bert_ids, bert_masks, bert_segments])

    # [2*batch_size*num_split, BERT_SEQ_LEN, 768] -> [2*batch_size, num_split, BERT_SEQ_LEN, 768]
    sequence_output = tf.reshape(sequence_output, [-1, num_split, BERT_SEQUENCE_LEN, 768])
    # sequence_output = layers.Reshape([-1, num_split, BERT_SEQUENCE_LEN, 768])(sequence_output)

    # [2*batch_size, num_split, BERT_SEQ_LEN, 768] -> [2*batch_size, num_split, 768]
    split_avg_pooled = layers.Dropout(0.2)(tf.reduce_mean(sequence_output, axis=2))
    split_max_pooled = layers.Dropout(0.2)(tf.reduce_max(sequence_output, axis=2))

    # [2*batch_size, num_split, 768] -> [2*batch_size, num_split*768]
    avg_pooled = tf.reshape(split_avg_pooled, [-1, num_split * 768])
    # avg_pooled = layers.Reshape([-1, num_split * 768])(split_avg_pooled)
    max_pooled = tf.reshape(split_max_pooled, [-1, num_split * 768])
    # max_pooled = layers.Reshape([-1, num_split * 768])(split_max_pooled)

    # [2*batch_size, num_split*768] -> [2, batch_size, num_split*768]
    split_avg_output = tf.reshape(avg_pooled, [2, -1, num_split * 768])
    # split_avg_output = layers.Reshape([2, -1, num_split * 768])(avg_pooled)
    split_max_output = tf.reshape(max_pooled, [2, -1, num_split * 768])
    # split_max_output = layers.Reshape([2, -1, num_split * 768])(max_pooled)

    # [2, batch_size, num_split*768] -> 2 x [batch_size, num_split*768]
    question_avg_pooled_output, qa_avg_pooled_output = tf.unstack(split_avg_output, axis=0)
    # question_avg_pooled_output, qa_avg_pooled_output = split_avg_output[0], split_avg_output[1]
    question_max_pooled_output, qa_max_pooled_output = tf.unstack(split_max_output, axis=0)
    # question_max_pooled_output, qa_max_pooled_output = split_max_output[0], split_max_output[1]

    # 2 x [batch_size, num_split*768] -> [batch_size, 2*num_split*768]
    question_pooled_output = layers.Concatenate()([question_avg_pooled_output, question_max_pooled_output])
    qa_pooled_output = layers.Concatenate()([qa_avg_pooled_output, qa_max_pooled_output])

    # [batch_size, 2*num_split*768] -> [batch_size, output_size]
    question_out = layers.Dense(21, activation="sigmoid", name="question_output")(question_pooled_output)
    qa_out = layers.Dense(9, activation="sigmoid", name="qa_output")(qa_pooled_output)

    out = layers.Concatenate()([question_out, qa_out])

    model = tf.keras.models.Model(
        inputs=[question_ids, question_masks, question_segments, qa_ids, qa_masks, qa_segments],
        outputs=out)

    return model


def train_and_predict(model, train_data, valid_data, test_data,
                      learning_rate, epochs, batch_size, loss_function, fold):
    custom_callback = CustomCallback(
        valid_data=(valid_data[0], valid_data[1]),
        test_data=test_data,
        batch_size=batch_size,
        fold=fold)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = AccumOptimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate), accumulation_step, name='AccOp')
    model.compile(loss=loss_function, optimizer=optimizer)
    model.fit(train_data[0], train_data[1], epochs=epochs, verbose=1, batch_size=batch_size,
              callbacks=[custom_callback])

    return custom_callback


# multi bert 输出
outputs = compute_output_arrays(df_train, question_categories + qa_categories)

# 计算multi bert 输入
inputs = []
inputs += compute_split_input_arrays(df_train, question_input_categories, tokenizer, BERT_SEQUENCE_LEN, num_split)
inputs += compute_split_input_arrays(df_train, qa_input_categories, tokenizer, BERT_SEQUENCE_LEN, num_split)

test_inputs = []
test_inputs += compute_split_input_arrays(df_test, question_input_categories, tokenizer, BERT_SEQUENCE_LEN, num_split)
test_inputs += compute_split_input_arrays(df_test, qa_input_categories, tokenizer, BERT_SEQUENCE_LEN, num_split)

kf = MultilabelStratifiedKFold(n_splits=num_folds).split(df_train, outputs)

histories = []
cv_scores = []
pred_train = np.zeros([df_train.shape[0], len(output_categories)])
train_rho_values = []
test_predictions = []

for fold, (train_idx, valid_idx) in enumerate(kf):
    if fold < start_fold:
        continue
    if fold >= start_fold + actual_fold_num:
        break

    model = multi_split_bert_model()
    if fold == start_fold:
        print(model.summary())

    train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
    train_outputs = outputs[train_idx]

    valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
    valid_outputs = outputs[valid_idx]

    if not SAVE_MODELS:
        fold = None
    tmp_test_inputs = [np.array(arr) for arr in test_inputs]
    history = train_and_predict(model,
                                train_data=(train_inputs, train_outputs),
                                valid_data=(valid_inputs, valid_outputs),
                                test_data=tmp_test_inputs,
                                learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                                loss_function='binary_crossentropy', fold=fold)
    pred_train[valid_idx, :] = np.array(history.best_valid_predictions)  # return a copy, so we can delete 'history'
    test_predictions.append(np.array(history.test_predictions))  # use a copy, so we can delete 'history'
    train_rho_values.append(np.array(history.train_rho_values))
    cv_scores.append(history.rho_value)  # rho_value is only a scalar, not an object, so no need to copy it

    # collect memory
    del train_inputs, train_outputs, valid_inputs, valid_outputs, tmp_test_inputs, history, model
    K.clear_session()
    [gc.collect() for _ in range(15)]

print(f"cv scores: {cv_scores}")

if not SAVE_MODELS:
    ps = tf.nn.softmax(train_rho_values).numpy()
    test_predictions = [np.average(test_predictions[i], axis=0, weights=ps[i]) for i in range(len(test_predictions))]
    test_predictions = np.mean(test_predictions, axis=0)
    df_sub.iloc[:, 1:] = test_predictions
    df_sub.to_csv('submission.csv', index=False)
else:
    if not os.path.exists('models/'):
        os.mkdir('models/')
    train_rho_df = pd.DataFrame(np.array(train_rho_values), columns=[f'epoch-{x}' for x in range(epochs)])
    train_rho_df.to_csv(f'models/train-rho-start-{start_fold}.csv', index=False)
    pred_train_df = pd.DataFrame(pred_train, columns=output_categories)
    pred_train_df.to_csv(f'models/pred-train-{start_fold}.csv', index=False)
