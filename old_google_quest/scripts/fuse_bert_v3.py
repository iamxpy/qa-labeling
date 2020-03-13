#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from stratifiers import MultilabelStratifiedKFold
import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import bert_tokenization as tokenization
import tensorflow.keras.backend as K
import gc
import os
from scipy.stats import spearmanr
from bert_utils import compute_1sen_input_arrays, compute_output_arrays

np.set_printoptions(suppress=True)

SAVE_MODELS = False
learning_rate = 3e-5
epochs = 10
batch_size = 4
num_folds = 10
start_fold = 0
actual_fold_num = 5
seed = 2019

PATH = '../input/google-quest-challenge/'
BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
tokenizer = tokenization.FullTokenizer(BERT_PATH + '/assets/vocab.txt', True)
MAX_SEQUENCE_LENGTH = 512

df_train = pd.read_csv(PATH + 'train.csv')
df_test = pd.read_csv(PATH + 'test.csv')
df_sub = pd.read_csv(PATH + 'sample_submission.csv')
print('train shape =', df_train.shape)
print('test shape =', df_test.shape)

original_output_categories = list(df_train.columns[11:])

question_input_categories = list(df_train.columns[[1, 2]])
answer_input_categories = list(df_train.columns[[5]])
print('\noutput categories:\n\t', original_output_categories)

question_out_categories = list(df_train.columns[11:32])  # 21 categories
answer_out_categories = list(df_train.columns[[33, 37, 38, 39, 40]])  # 5 categories
qa_out_categories = list(df_train.columns[[32, 34, 35, 36]])  # 4 categories

fused_output_categories = question_out_categories + answer_out_categories + qa_out_categories


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


seed_everything(seed)


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, test_data, batch_size=16, fold=None):
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
        # check whether to save model and update best rho value
        if rho_val >= self.rho_value:
            self.rho_value = rho_val
            self.best_valid_predictions = val_pred

        print("\nvalidation rho: %.4f" % rho_val)


def fake_bert_for_debug(seq_len=512):
    def fake_bert(input_list):
        emb_word = tf.keras.layers.Embedding(50000, 768, input_length=seq_len, trainable=False)(input_list[0])
        return None, emb_word

    return fake_bert


def multi_bert_model():
    input_question_ids = tf.keras.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_question_ids')
    input_question_masks = tf.keras.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_question_masks')
    input_question_segments = tf.keras.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_question_segments')

    input_answer_ids = tf.keras.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_answer_ids')
    input_answer_masks = tf.keras.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_answer_masks')
    input_answer_segments = tf.keras.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_answer_segments')

    # TODO DEBUG in CPU
    # question_bert_layer = fake_bert_for_debug()
    question_bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)

    # TODO debug in cpu
    # answer_bert_layer = fake_bert_for_debug()
    answer_bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)

    # word embedding
    _, question_sequence_output = question_bert_layer(
        [input_question_ids, input_question_masks, input_question_segments])
    _, answer_sequence_output = answer_bert_layer([input_answer_ids, input_answer_masks, input_answer_segments])

    # sentence embedding
    question_x = tf.keras.layers.Dropout(0.2)(tf.keras.layers.GlobalAveragePooling1D()(question_sequence_output))
    answer_x = tf.keras.layers.Dropout(0.2)(tf.keras.layers.GlobalAveragePooling1D()(answer_sequence_output))
    qa_x = tf.keras.layers.Concatenate()([question_x, answer_x])

    # MLP
    question_out = tf.keras.layers.Dense(21, activation="sigmoid", name="question_output")(question_x)
    answer_out = tf.keras.layers.Dense(5, activation="sigmoid", name="answer_output")(answer_x)
    qa_out = tf.keras.layers.Dense(4, activation="sigmoid", name="qa_output")(qa_x)

    # concat 3 output
    out = tf.keras.layers.Concatenate()([question_out, answer_out, qa_out])

    model = tf.keras.models.Model(
        inputs=[input_question_ids, input_question_masks, input_question_segments,
                input_answer_ids, input_answer_masks, input_answer_segments], outputs=out)

    return model


def train_and_predict(model, train_data, valid_data, test_data,
                      learning_rate, epochs, batch_size, loss_function, fold):
    custom_callback = CustomCallback(
        valid_data=(valid_data[0], valid_data[1]),
        test_data=test_data,
        batch_size=batch_size,
        fold=fold)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer)
    model.fit(train_data[0], train_data[1], epochs=epochs, verbose=1,
              batch_size=batch_size, callbacks=[custom_callback])

    return custom_callback


# multi bert 输出
outputs = compute_output_arrays(df_train, fused_output_categories)

# 计算multi bert 输入
inputs = []
inputs += compute_1sen_input_arrays(df_train, question_input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
inputs += compute_1sen_input_arrays(df_train, answer_input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

test_inputs = []
test_inputs += compute_1sen_input_arrays(df_test, question_input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs += compute_1sen_input_arrays(df_test, answer_input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

kf = MultilabelStratifiedKFold(n_splits=num_folds, random_state=seed).split(df_train, outputs)

histories = []
cv_scores = []
pred_train = np.zeros([df_train.shape[0], len(fused_output_categories)])
train_rho_values = []
test_predictions = []

for fold, (train_idx, valid_idx) in enumerate(kf):
    if fold < start_fold:
        continue
    if fold >= start_fold + actual_fold_num:
        break

    model = multi_bert_model()
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
    # pred_test += np.array(history.test_predictions)
    train_rho_values.append(np.array(history.train_rho_values))
    cv_scores.append(history.rho_value)  # rho_value is only a scalar, not an object, so need not to copy it

    # collect memory
    del train_inputs, train_outputs, valid_inputs, valid_outputs, tmp_test_inputs, history, model
    K.clear_session()
    [gc.collect() for _ in range(15)]

print(f"cv scores: {cv_scores}")

if not SAVE_MODELS:
    ps = tf.nn.softmax(train_rho_values).numpy()
    test_predictions = [np.average(test_predictions[i], axis=0, weights=ps[i]) for i in range(len(test_predictions))]
    test_predictions = np.mean(test_predictions, axis=0)

    # convert to original output
    test_predictions = pd.DataFrame(test_predictions, columns=fused_output_categories)
    test_predictions = test_predictions[original_output_categories]

    df_sub.iloc[:, 1:] = test_predictions
    df_sub.to_csv('submission.csv', index=False)
else:
    if not os.path.exists('models/'):
        os.mkdir('models/')
    train_rho_df = pd.DataFrame(np.array(train_rho_values), columns=[f'epoch-{x}' for x in range(epochs)])
    train_rho_df.to_csv(f'models/train-rho-start-{start_fold}.csv', index=False)

    # convert to original output
    pred_train_df = pd.DataFrame(pred_train, columns=fused_output_categories)
    pred_train_df = pred_train_df[original_output_categories]

    pred_train_df.to_csv(f'models/pred-train-{start_fold}.csv', index=False)
