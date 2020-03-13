import pandas as pd
import tensorflow_hub as hub
import tensorflow.keras.layers as layers
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import gc
import random
import os
import bert_tokenization as tokenization
from bert_utils import compute_3sen_input_arrays_new, compute_1sen_input_arrays

df_train = pd.read_csv('../input/google-quest-challenge/train.csv')
df_test = pd.read_csv('../input/google-quest-challenge/test.csv')
df_sub = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
tokenizer = tokenization.FullTokenizer(BERT_PATH + '/assets/vocab.txt', True)
input_categories = list(df_test.columns[[1, 2, 5]])

max_seq_len = 512
batch_size = 16
seed = 2019


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


def fix(candidate, expected_values):
    gaps = np.abs(expected_values - candidate)
    fixed = expected_values[np.argmin(gaps)]
    return fixed


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

values = []
denominator = [6, 9, 10, 15]
for f in denominator:
    for i in range(f):
        a = i / f
        if a not in values:
            values.append(a)
values = np.array(values)

# 加载测试数据
# test_inputs = compute_input_arays(df_test, input_categories, tokenizer, max_seq_len)
question_input_categories = list(df_train.columns[[1, 2]])
qa_input_categories = list(df_train.columns[[1, 2, 5]])
test_inputs = []
test_inputs += compute_1sen_input_arrays(df_test, question_input_categories, tokenizer, max_seq_len)
test_inputs += compute_3sen_input_arrays_new(df_test, qa_input_categories, tokenizer, max_seq_len)

# ms = [(0, 2), (2, 2), (3, 3), (5, 2), (8, 4)]
# folds = [0, 2, 3, 5, 8]
folds = list(range(10))
scores = pd.read_csv('models/scores.csv').iloc[0, folds]

test_predictions = []
# for fold, epoch in ms:
for fold in folds:
    model = multi_bert_model()
    # 加载模型参数
    p = f'models/fold-{fold}-best.h5'
    print(f"Loading model from: {p}")
    model.load_weights(p)
    print("Predicting ......")
    tmp_test_inputs = [np.array(arr) for arr in test_inputs]
    pred = model.predict(tmp_test_inputs, batch_size=batch_size)
    test_predictions.append(np.array(pred))
    del p, tmp_test_inputs, pred, model
    K.clear_session()
    [gc.collect() for _ in range(15)]

ps = tf.nn.softmax(scores).numpy()

pred = np.average(test_predictions, axis=0, weights=ps)
#
# for i in range(len(pred)):
#     for j in range(30):
#         pred[i][j] = fix(pred[i][j], values) + 1e-7
#
# fixed_columns = 0
# for i in range(30):
#     if np.all(pred[:, i] == pred[0, i]):
#         pred[0, i] += 1e-7
#         fixed_columns += 1
#
# print(f"{fixed_columns} columns fixed.")

df_sub.iloc[:, 1:] = pred
df_sub.to_csv('submission.csv', index=False)
print("Done!")
