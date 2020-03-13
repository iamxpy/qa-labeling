#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
num_sample = 1000
batch_size = 10
max_seq_len = 512
num_class = 30
vocab_num = 30000
epochs = 100
learning_rate = 1e-5

import tensorflow as tf
import tensorflow_hub as hub


def get_intermediate_layer(last_layer, total_layers, desired_layer):
    print("Last layer name: ", last_layer.name)
    intermediate_layer_name = last_layer.name.replace(str(total_layers + 1),
                                                      str(desired_layer + 1))
    print("Intermediate layer name: ", intermediate_layer_name)
    return tf.get_default_graph().get_tensor_by_name(intermediate_layer_name)


input_ids = tf.zeros([1, 512], tf.int32)
input_mask = tf.zeros([1, 512], tf.int32)
segment_ids = tf.zeros([1, 512], tf.int32)
bert_inputs = dict(
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids)

module = hub.KerasLayer("/home/xiepengyu/google_quest/input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12/")
# token_signature = module.signatures["tokens"]  # this will give me KeyError
# token_signature = module.signatures["serving_default"]
# module_input = dict(
#     input_word_ids=tf.constant(3, shape=[1, 4]),
#     input_mask=tf.constant(1, shape=[1, 4]),
#     input_type_ids=tf.constant(4, shape=[1, 4]),
# )
# output = token_signature(**module_input)
# print(output)

output = module([input_ids, input_mask, segment_ids])

layer_6 = get_intermediate_layer(
    last_layer=output[1],
    total_layers=12,
    desired_layer=6)

print(layer_6)

# def bert_model():
#     input_ids = tf.keras.Input((max_seq_len,), dtype=tf.int32, name='input_ids')
#     input_masks = tf.keras.Input((max_seq_len,), dtype=tf.int32, name='input_masks')
#     input_segments = tf.keras.Input((max_seq_len,), dtype=tf.int32, name='input_segments')
#
#     bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)
#
#     pooled_output, sequence_output = bert_layer([input_ids, input_masks, input_segments])
#
#     out = tf.keras.layers.Dense(num_class, activation="sigmoid", name="dense_output")(pooled_output)
#
#     model = tf.keras.models.Model(inputs=[input_ids, input_masks, input_segments], outputs=out)
#
#     return model
#
#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)
#
# outputs = np.random.randn(num_sample, num_class)
#
# inputs = [np.random.randint(vocab_num, size=(num_sample, max_seq_len), dtype=np.int32),  # ids
#           np.ones((num_sample, max_seq_len), dtype=np.int32),  # masks
#           np.zeros((num_sample, max_seq_len), dtype=np.int32)]  # segments
#
# model = bert_model()
# print(model.summary())
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# model.compile(loss='binary_crossentropy', optimizer=optimizer)  # multi-lebel task
# model.fit(inputs, outputs, epochs=epochs, verbose=1, batch_size=batch_size)
