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
from scipy.stats import spearmanr
from math import floor, ceil
from bert_utils import compute_3sen_input_arrays_with_translate, compute_1sen_input_arrays_with_translate, \
    compute_3sen_input_arrays, compute_1sen_input_arrays, compute_output_arrays

np.set_printoptions(suppress=True)

# SAVE_MODELS = True
learning_rate = 3e-5
epochs = 1
aug_times = 4
batch_size = 4
num_folds = 10
start_fold = 0
actual_fold_num = 1
seed = 2019

PATH = '../input/google-quest-challenge/'
BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
tokenizer = tokenization.FullTokenizer(BERT_PATH + '/assets/vocab.txt', True)
max_seq_len = 512

df_train = pd.read_csv(PATH + 'train_trans_more.csv')
# df_test = pd.read_csv(PATH + 'test.csv')
print('train shape =', df_train.shape)
# print('test shape =', df_test.shape)

output_categories = list(df_train.columns[11:])
question_input_categories = list(df_train.columns[[1, 2]])
qa_input_categories = list(df_train.columns[[1, 2, 5]])
print('\noutput categories:\n\t', output_categories)

# question_categories = list(df_train.columns[11:32])  # 21 categories
# qa_categories = list(df_train.columns[32:])  # 9 categories
question_categories = ['question_asker_intent_understanding',
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
                       'question_well_written']
qa_categories = ['answer_helpful',
                 'answer_level_of_information',
                 'answer_plausible',
                 'answer_relevance',
                 'answer_satisfaction',
                 'answer_type_instructions',
                 'answer_type_procedure',
                 'answer_type_reason_explanation',
                 'answer_well_written']


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
    return np.mean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, batch_size=16, fold=None):
        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.batch_size = batch_size
        self.fold = fold
        self.val_loss = float('inf')
        self.rho_value = -1  # record the best rho for report
        self.valid_predictions = []
        # self.val_rho_values = []

    def on_epoch_end(self, epoch, logs={}):
        val_pred = self.model.predict(self.valid_inputs, batch_size=self.batch_size)
        self.valid_predictions.append(val_pred)
        rho_val = compute_spearmanr(
            self.valid_outputs, np.average(self.valid_predictions, axis=0))
        # self.val_rho_values.append(rho_val)
        # if self.fold is not None:
        #     if not os.path.exists('models/'):
        #         os.mkdir('models/')
        #     self.model.save_weights(f'models/fold-{fold}-epoch-{epoch}.h5')  # 一共保存fold * epoch个模型
        # check whether to save model and update best rho value
        # todo delete
        rho_val = 0 if np.isnan(rho_val) else rho_val
        if rho_val >= self.rho_value:
            self.rho_value = rho_val
            if self.fold is not None:
                if not os.path.exists('models/'):
                    os.mkdir('models/')
                self.model.save_weights(f'models/fold-{fold}-best.h5')

        print("\nvalidation rho: %.4f" % rho_val)


def multi_bert_model():
    input_question_ids = tf.keras.Input((max_seq_len,), dtype=tf.int32, name='input_question_ids')
    input_question_masks = tf.keras.Input((max_seq_len,), dtype=tf.int32, name='input_question_masks')
    input_question_segments = tf.keras.Input((max_seq_len,), dtype=tf.int32, name='input_question_segments')

    input_qa_ids = tf.keras.Input((max_seq_len,), dtype=tf.int32, name='input_qa_ids')
    input_qa_masks = tf.keras.Input((max_seq_len,), dtype=tf.int32, name='input_qa_masks')
    input_qa_segments = tf.keras.Input((max_seq_len,), dtype=tf.int32, name='input_qa_segments')

    question_bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)

    qa_bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)
    _, question_sequence_output = question_bert_layer(
        [input_question_ids, input_question_masks, input_question_segments])
    _, qa_sequence_output = qa_bert_layer([input_qa_ids, input_qa_masks, input_qa_segments])

    question_x = layers.Dropout(0.2)(layers.GlobalAveragePooling1D()(question_sequence_output))
    qa_x = layers.Dropout(0.2)(layers.GlobalAveragePooling1D()(qa_sequence_output))
    question_logits = layers.Dense(21, name="question_logits")(question_x)
    qa_logits = layers.Dense(9, name="qa_logits")(qa_x)

    # when using mixed precision, regardless of what your model ends in, make sure the output is float32.
    # we need to set the dtype of last layer (before loss computation) to float32
    question_out = layers.Activation('sigmoid', dtype='float32', name='question_output')(question_logits)
    qa_out = layers.Activation('sigmoid', dtype='float32', name='qa_output')(qa_logits)
    # concat 3 bert output
    # out = tf.keras.layers.Concatenate()([question_out, answer_out, qa_out])
    out = layers.Concatenate()([question_out, qa_out])

    model = tf.keras.models.Model(
        inputs=[input_question_ids, input_question_masks, input_question_segments,
                # input_answer_ids, input_answer_masks, input_answer_segments,
                input_qa_ids, input_qa_masks, input_qa_segments], outputs=out)

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


# 计算multi bert 输入
# inputs_ori = []
# inputs_ori += compute_1sen_input_arrays(df_train, question_input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
# inputs_ori += compute_3sen_input_arrays(df_train, qa_input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
#
# inputs_aug = []
# inputs_aug += compute_1sen_input_arrays_with_translate(df_train, question_input_categories, tokenizer,
#                                                        MAX_SEQUENCE_LENGTH, True)
# inputs_aug += compute_3sen_input_arrays_with_translate(df_train, qa_input_categories, tokenizer, MAX_SEQUENCE_LENGTH,
#                                                        True)

# multi bert 输出
outputs = compute_output_arrays(df_train, question_categories + qa_categories)

kf = MultilabelStratifiedKFold(n_splits=num_folds).split(df_train, outputs)

histories = []
cv_scores = []
# pred_train = np.zeros([df_train.shape[0], len(output_categories)])
# val_rho_values = []
# test_predictions = []

# for fold, (train_idx, valid_idx) in enumerate(gkf):
for fold, (train_idx, valid_idx) in enumerate(kf):
    if fold < start_fold:
        continue
    if fold >= start_fold + actual_fold_num:
        break

    model = multi_bert_model()
    # train_inputs = [inputs_aug[i][train_idx] for i in range(len(inputs_aug))]
    train_set = df_train.iloc[train_idx]
    train_inputs = []
    tmp = [compute_1sen_input_arrays_with_translate(train_set, question_input_categories, tokenizer,
                                                    max_seq_len, True) for _ in range(aug_times)]
    train_inputs += [np.concatenate([tmp[j][i] for j in range(aug_times)]) for i in range(3)]  # 感觉自己有点秀
    # train_inputs += compute_1sen_input_arrays_with_translate(train_set, question_input_categories, tokenizer,
    #                                                          max_seq_len, True)
    tmp = [compute_3sen_input_arrays_with_translate(train_set, qa_input_categories, tokenizer,
                                                    max_seq_len, True) for _ in range(aug_times)]
    train_inputs += [np.concatenate([tmp[j][i] for j in range(aug_times)]) for i in range(3)]  # 感觉自己有点秀
    # train_inputs += compute_3sen_input_arrays_with_translate(train_set, qa_input_categories, tokenizer,
    #                                                          max_seq_len, True)
    # train_outputs = outputs[train_idx]
    tmp = [compute_output_arrays(train_set, question_categories + qa_categories) for _ in range(aug_times)]
    train_outputs = np.concatenate(tmp, axis=0)
    # train_outputs = compute_output_arrays(train_set, question_categories + qa_categories)

    valid_set = df_train.iloc[valid_idx]
    valid_inputs = []
    valid_inputs += compute_1sen_input_arrays(valid_set, question_input_categories, tokenizer, max_seq_len)
    valid_inputs += compute_3sen_input_arrays(valid_set, qa_input_categories, tokenizer, max_seq_len)
    # valid_inputs = [inputs_ori[i][valid_idx] for i in range(len(inputs_ori))]
    # valid_outputs = outputs[valid_idx]
    valid_outputs = compute_output_arrays(valid_set, question_categories + qa_categories)

    history = train_and_predict(model,
                                train_data=(train_inputs, train_outputs),
                                valid_data=(valid_inputs, valid_outputs),
                                learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                                loss_function='binary_crossentropy', fold=fold)
    # val_rho_values.append(np.array(history.val_rho_values))
    cv_scores.append(history.rho_value)  # rho_value is only a scalar, not an object, so need not to copy it

    # collect memory
    del tmp, train_inputs, train_outputs, valid_inputs, valid_outputs, history, model
    K.clear_session()
    [gc.collect() for _ in range(15)]

print(f"cv scores: {cv_scores}")

if not os.path.exists('models/'):
    os.mkdir('models/')
# train_rho_df = pd.DataFrame(np.array(val_rho_values), columns=[f'epoch-{x}' for x in range(epochs)])
train_rho_df = pd.DataFrame([cv_scores], columns=[f'fold-{x}' for x in range(start_fold, start_fold + actual_fold_num)])
train_rho_df.to_csv(f'models/train-rho-start-{start_fold}.csv', index=False)
