#!/usr/bin/env python
# coding: utf-8
import argparse

import pandas as pd
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
from bert_utils import compute_3sen_input_arrays, compute_output_arrays
from math import floor, ceil
from bert_models import BertLinear


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


def on_epoch_end(self, epoch, logs={}):
    val_pred = self.model.predict(self.valid_inputs, batch_size=self.batch_size)
    self.valid_predictions.append(val_pred)
    rho_val = compute_spearmanr(
        # self.valid_outputs, np.average(self.valid_predictions, axis=0))
        self.valid_outputs, val_pred)
    self.train_rho_values.append(rho_val)
    self.test_predictions.append(self.model.predict(self.test_inputs, batch_size=self.batch_size))
    # check whether to save model and update best rho value
    if rho_val >= self.rho_value:
        self.rho_value = rho_val
        self.best_valid_predictions = val_pred
        if self.fold is not None:
            if not os.path.exists('models/'):
                os.mkdir('models/')
            self.model.save_weights(f'models/bert-base-fold-{fold}.h5')
    print("\nvalidation rho: %.4f" % rho_val)


def train_and_predict(model, args, train_data, valid_data):
    rho_values = []
    for epoch in range(args.epochs):
        print(f"================ epoch: {epoch} ==================")
        for i, batch in enumerate(next_batch(train_data[0], train_data[1], args.train_batch_size)):
            logits, loss, summary, current_step, _ = model.predict(batch)
            # todo logging
            model.train_writer.add_summary(summary, current_step)
            if current_step > 0 and current_step % args.eval_every == 0:
                eval_losses = []
                # eval_reg_losses = []
                eval_rhos = []
                for eval_batch in next_batch(valid_data[0], valid_data[1], args.eval_batch_size):
                    # preds, loss, reg_loss = model.predict(batch, is_training=False)
                    preds, loss= model.predict(batch, is_training=False)
                    eval_losses.append(loss)
                    # eval_reg_losses.append(reg_loss)
                    rho = compute_spearmanr(eval_batch[1], preds)
                    eval_rhos.append(rho)
                mean_loss = np.mean(eval_losses)
                # mean_reg_loss = np.mean(eval_reg_losses)
                mean_rho = np.mean(eval_rhos)
                rho_values.append(mean_rho)
                add_summary(model.eval_writer, current_step, "losses/total", mean_loss)
                # add_summary(model.eval_writer, current_step, "losses/reg", mean_reg_loss)
                add_summary(model.eval_writer, current_step, "rho", mean_rho)
                # todo logging
                print(
                    "=" * 20 + f"  epoch: {epoch}, batch: {i}, loss: {round(float(mean_loss), 4)}, rho: {round(float(mean_rho), 4)}  " + "=" * 20)
    return rho_values


def add_summary(writer, step, tag, value):
    '''
    将单个指标写入tensorboard
    '''
    stat = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(stat, step)


def next_batch(x_list, y, batch_size, shuffle=False):
    if shuffle:
        # 构造索引, 然后对索引进行shuffle
        perm = np.arange(len(y))
        np.random.shuffle(perm)
        # 然后根据打乱的索引重排x和y
        x_list = [x[perm] for x in x_list]
        y = y[perm]
    num_batches = int(ceil(len(y) // batch_size))
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_x = [x[start: end] for x in x_list]
        batch_y = y[start: end]
        yield batch_x, batch_y


def main():
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.learning_rate = 3e-5
    args.epochs = 10
    args.train_batch_size = 8
    args.eval_batch_size = 8
    args.eval_every = 300
    args.dropout_rate = 0.2
    args.max_seq_len = 512
    args.max_grad_norm = 5.0
    num_folds = 5
    actual_folds = 5
    seed = 2019

    PATH = '../input/google-quest-challenge/'
    BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
    tokenizer = tokenization.FullTokenizer(BERT_PATH + '/assets/vocab.txt', True)

    seed_everything(seed)

    # todo delete
    df_train = pd.read_csv(PATH + 'train.csv').head(200)
    # df_train = pd.read_csv(PATH + 'train.csv')
    print('train shape =', df_train.shape)

    output_categories = list(df_train.columns[11:])
    input_categories = list(df_train.columns[[1, 2, 5]])
    print('\noutput categories:\n\t', output_categories)
    print('\ninput categories:\n\t', input_categories)

    inputs = compute_3sen_input_arrays(df_train, input_categories, tokenizer, args.max_seq_len)
    outputs = compute_output_arrays(df_train, output_categories)

    kf = MultilabelStratifiedKFold(n_splits=num_folds).split(df_train, outputs)

    val_rho_values = []
    cv_scores = []

    for fold, (train_idx, valid_idx) in enumerate(kf):

        if fold >= actual_folds:
            break

        model = BertLinear(args)

        train_inputs = [inputs[i][train_idx] for i in range(3)]
        train_outputs = outputs[train_idx]

        valid_inputs = [inputs[i][valid_idx] for i in range(3)]
        valid_outputs = outputs[valid_idx]

        rhos = train_and_predict(model, args,
                                 train_data=(train_inputs, train_outputs),
                                 valid_data=(valid_inputs, valid_outputs))
        val_rho_values.append(np.array(rhos))
        cv_scores.append(max(rhos))

        # collect memory
        del train_inputs, train_outputs, valid_inputs, valid_outputs, rhos, model
        K.clear_session()
        [gc.collect() for _ in range(15)]

    print(f"cv scores: {cv_scores}")

    if not os.path.exists('../output/models/'):
        os.makedirs('../output/models/')
    train_rho_df = pd.DataFrame(np.array(val_rho_values), columns=[f'epoch-{x}' for x in range(args.epochs)])
    train_rho_df.to_csv(f'../output/models/train-rho.csv', index=False)


if __name__ == '__main__':
    main()
