#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import tensorflow_hub as hub
import tensorflow.keras.mixed_precision.experimental as mixed_precision
import tensorflow.keras.layers as layers
from stratifiers import MultilabelStratifiedKFold
import random
import numpy as np
import tensorflow as tf
import bert_tokenization as tokenization
import tensorflow.keras.backend as K
import gc
import os
import time
from scipy.stats import spearmanr
from math import floor, ceil
from adjust import *
from bert_utils import compute_3sen_input_arrays_new, compute_1sen_input_arrays, compute_output_arrays

np.set_printoptions(suppress=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

learning_rate = 3e-5
epochs = 7
train_bsz = 1
eval_every = 200
eval_bsz = 4
num_folds = 5
start_fold = 0
actual_fold_num = 1
seed = 2019
output_path = f'output-nf-{num_folds}-{start_fold}-{start_fold + actual_fold_num - 1}-epochs-{epochs}-lr-{learning_rate}-tbsz-{train_bsz}-ebsz-{eval_bsz}'
if not os.path.exists(output_path):
    os.makedirs(output_path)

PATH = '../input/google-quest-challenge/'
# BERT_PATH = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
BERT_PATH = '../input/bert-base-from-tfhub/bert-base-uncased'
# BERT_PATH = '../input/bert-base-from-tfhub/bert_en_wwm_uncased_L-24_H-1024_A-16'
tokenizer = tokenization.FullTokenizer(BERT_PATH + '/assets/vocab.txt', True)
MAX_SEQUENCE_LENGTH = 512

df_train = pd.read_csv(PATH + 'train.csv')
print('train shape =', df_train.shape)

output_categories = list(df_train.columns[11:])
question_input_categories = list(df_train.columns[[1, 2]])
qa_input_categories = list(df_train.columns[[1, 2, 5]])
print('\noutput categories:\n\t', output_categories)

question_categories = list(df_train.columns[11:32])  # 21 categories
qa_categories = list(df_train.columns[32:])  # 9 categories


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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
    return float(np.mean(rhos))


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, output_path, eval_every=500, eval_batch=16, fold=None):
        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.eval_batch = eval_batch
        self.fold = fold
        self.output_path = output_path
        self.val_loss = float('inf')
        self.rho_value = -1  # record the best rho for report
        self.eval_every = eval_every

    def on_train_begin(self, logs={}):
        self.start_time = time.time()
        self.epoch_steps = ceil(self.params['samples'] / self.params['batch_size'])
        print("step  epoch  train_loss | eval_loss  rho  post_rho  other_rho  align_rho | time")
        print('-' * 100)

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):

        if batch > 0 and batch % self.eval_every == 0:
            val_pred = self.model.predict(self.valid_inputs, batch_size=self.eval_batch)
            val_loss = np.mean(tf.keras.losses.binary_crossentropy(self.valid_outputs, val_pred)).item()

            post_pred = val_pred.copy()
            other_pred = val_pred.copy()
            align_pred = val_pred.copy()

            post_pred = post_deal_result(post_pred)
            other_pred = other_deal_result(other_pred)
            align_pred = align_result(align_pred)

            rho = compute_spearmanr(self.valid_outputs, val_pred)
            post_rho = compute_spearmanr(self.valid_outputs, post_pred)
            other_rho = compute_spearmanr(self.valid_outputs, other_pred)
            align_rho = compute_spearmanr(self.valid_outputs, align_pred)

            # check whether to save model and update best rho value
            if rho > self.rho_value:
                self.rho_value = rho
                self.model.save_weights(f'{self.output_path}/fold-{fold}-best.h5')

            elapsed_time = time.time() - self.start_time
            print("%.1f  %.1f  %.4f | %.4f  %.4f  %.4f  %.4f  %.4f | %.4f seconds" % (
                batch / 1e3, batch / self.epoch_steps, logs.get('loss'), val_loss, rho, post_rho, other_rho, align_rho,
                elapsed_time))
            gc.collect()


def multi_bert_model():
    input_question_ids = tf.keras.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_question_ids')
    input_question_masks = tf.keras.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_question_masks')
    input_question_segments = tf.keras.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_question_segments')

    input_qa_ids = tf.keras.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_qa_ids')
    input_qa_masks = tf.keras.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_qa_masks')
    input_qa_segments = tf.keras.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_qa_segments')

    question_bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)

    qa_bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)

    _, question_sequence_output = question_bert_layer(
        [input_question_ids, input_question_masks, input_question_segments])
    _, qa_sequence_output = qa_bert_layer([input_qa_ids, input_qa_masks, input_qa_segments])

    question_avg_x = layers.Dropout(0.2)(tf.keras.layers.GlobalAveragePooling1D()(question_sequence_output))
    qa_avg_x = layers.Dropout(0.2)(tf.keras.layers.GlobalAveragePooling1D()(qa_sequence_output))

    question_max_x = layers.Dropout(0.2)(tf.keras.layers.GlobalMaxPooling1D()(question_sequence_output))
    qa_max_x = layers.Dropout(0.2)(tf.keras.layers.GlobalMaxPooling1D()(qa_sequence_output))

    question_x = layers.Concatenate()([question_avg_x, question_max_x])
    qa_x = layers.Concatenate()([qa_avg_x, qa_max_x])

    # when using mixed precision, regardless of what your model ends in, make sure the output is float32.
    question_out = tf.keras.layers.Dense(21, activation="sigmoid", dtype='float32', name="question_output")(question_x)
    qa_out = tf.keras.layers.Dense(9, activation="sigmoid", dtype='float32', name="qa_output")(qa_x)

    # concat 3 bert output
    # out = tf.keras.layers.Concatenate()([question_out, answer_out, qa_out])
    out = layers.Concatenate()([question_out, qa_out])

    model = tf.keras.models.Model(
        inputs=[input_question_ids, input_question_masks, input_question_segments,
                # input_answer_ids, input_answer_masks, input_answer_segments,
                input_qa_ids, input_qa_masks, input_qa_segments, ], outputs=out)

    return model


def store_cv(cvs, done_fold):
    train_rho_df = pd.DataFrame([cvs],
                                columns=[f'fold-{x}' for x in range(start_fold, start_fold + done_fold)])
    train_rho_df.to_csv(f'{output_path}/train-rho-start-{start_fold}.csv', index=False)


def train_and_predict(model, train_data, valid_data, output_path,
                      learning_rate, epochs, train_batch, eval_batch, loss_function, fold):
    custom_callback = CustomCallback(
        valid_data=(valid_data[0], valid_data[1]),
        output_path=output_path,
        eval_every=eval_every,
        eval_batch=eval_batch,
        fold=fold)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer)
    model.fit(train_data[0], train_data[1], epochs=epochs, verbose=2,
              batch_size=train_batch, callbacks=[custom_callback])

    return custom_callback


# we can use mixed precision with the following line
# mixed_precision.set_policy(mixed_precision.Policy('mixed_float16'))

# 计算multi bert 输入
inputs = []
inputs += compute_1sen_input_arrays(df_train, question_input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
inputs += compute_3sen_input_arrays_new(df_train, qa_input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

# multi bert 输出
outputs = compute_output_arrays(df_train, question_categories + qa_categories)

# todo 改为使用GroupKFold
kf = MultilabelStratifiedKFold(n_splits=num_folds).split(df_train, outputs)

cv_scores = []

# for fold, (train_idx, valid_idx) in enumerate(gkf):
for fold, (train_idx, valid_idx) in enumerate(kf):
    if fold < start_fold:
        continue
    if fold >= start_fold + actual_fold_num:
        break

    model = multi_bert_model()
    train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
    train_outputs = outputs[train_idx]

    valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
    valid_outputs = outputs[valid_idx]

    print(f"Train on fold {fold}.")

    history = train_and_predict(model,
                                train_data=(train_inputs, train_outputs),
                                valid_data=(valid_inputs, valid_outputs),
                                output_path=output_path,
                                learning_rate=learning_rate, epochs=epochs, train_batch=train_bsz, eval_batch=eval_bsz,
                                loss_function='binary_crossentropy', fold=fold)
    cv_scores.append(history.rho_value)  # rho_value is only a scalar, not an object, so need not to copy it
    store_cv(cv_scores, fold - start_fold + 1)

    # collect memory
    del train_inputs, train_outputs, valid_inputs, valid_outputs, history, model
    K.clear_session()
    [gc.collect() for _ in range(15)]

print(f"cv scores: {cv_scores}")
