import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import LSTMCell
import tensorflow_hub as hub


def print_vars(vars):
    total = 0
    for var in vars:
        print(var.name, var.get_shape())
        total += np.prod(var.get_shape().as_list())
    print(total)


class BertLinear(object):

    def __init__(self, args):
        super().__init__()
        self.max_grad_norm = args.max_grad_norm  # 5.0
        self.max_seq_len = args.max_seq_len  # 512
        self.dropout_rate = args.dropout_rate  # 0.2
        with tf.Graph().as_default():
            # regularizer = layers.l2_regularizer(0.)

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.global_step = tf.Variable(initial_value=0, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=3e-5)

            self.input_ids_ph = tf.placeholder(tf.int32, (None, self.max_seq_len), name="input_ids")
            self.input_masks_ph = tf.placeholder(tf.int32, (None, self.max_seq_len), name="input_masks")
            self.input_segments_ph = tf.placeholder(tf.int32, (None, self.max_seq_len), name="input_segments")
            self.dropout_rate_ph = tf.placeholder(tf.float32, name="dropout_rate")

            self.labels_ph = tf.placeholder(tf.float32, shape=(None, 30))

            bert_inputs = dict(
                input_ids=self.input_ids_ph,
                input_mask=self.input_masks_ph,
                segment_ids=self.input_segments_ph)
            bert_module = hub.Module("/home/xiepengyu/google_quest/input/bert-base-from-tfhub/bert/")
            bert_outputs = bert_module(
                inputs=bert_inputs,
                signature="tokens",
                as_dict=True)
            # sequence_output = self.get_intermediate_layer(
            #     last_layer=bert_outputs["sequence_output"],
            #     total_layers=12,
            #     desired_layer=12)
            sequence_output = bert_outputs["sequence_output"]
            # pooled_output = tf.nn.dropout(tf.reduce_mean(sequence_output, axis=1), rate=self.dropout_rate_ph)
            # pooled_output = tf.reduce_mean(sequence_output, axis=1)
            pooled_output = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
            fc_input = tf.keras.layers.Dropout(self.dropout_rate_ph)(pooled_output)
            # self.logits = layers.fully_connected(pooled_output, 30, weights_regularizer=regularizer)
            self.logits = tf.keras.layers.Dense(30, activation="sigmoid", name="dense_output")(fc_input)
            # self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.predictions = tf.math.sigmoid(self.logits)
            # total_loss = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels_ph, logits=self.logits))
            # self.loss = total_loss + self.reg_loss
            self.loss = tf.keras.losses.BinaryCrossentropy()(self.labels_ph, self.predictions)

            self.trainable_params = tf.trainable_variables()
            # gradients = tf.gradients(self.loss, self.trainable_params)
            # clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
            # 在optimizer.minimize()或optimizer.apply_gradient()时传入,则tf帮我们每更新一次梯度就使global_step+1
            # self.train_op = self.optimizer.apply_gradients(zip(clip_gradients, self.trainable_params),global_step=self.global_step)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            # print_vars(tf.trainable_variables())
            tf.summary.scalar('losses/total', self.loss)
            # tf.summary.scalar('losses/reg', self.reg_loss)
            if not os.path.exists('../output/'):
                os.makedirs('../output/')

            self.train_writer = tf.summary.FileWriter('../output/tensorboard/train/', self.sess.graph)
            self.eval_writer = tf.summary.FileWriter('../output/tensorboard/eval/', self.sess.graph)
            self.summary_op = tf.summary.merge_all()

    def get_intermediate_layer(self, last_layer, total_layers, desired_layer):
        intermediate_layer_name = last_layer.name.replace(str(total_layers + 1), str(desired_layer + 1))
        print("Intermediate layer name: ", intermediate_layer_name)
        return tf.get_default_graph().get_tensor_by_name(intermediate_layer_name)

    def get_feed_dict(self, batch, dropout_rate):
        (input_ids, input_masks, input_segments), labels = batch
        feed_dict = {self.input_ids_ph: input_ids, self.input_masks_ph: input_masks,
                     self.input_segments_ph: input_segments,
                     self.dropout_rate_ph: dropout_rate, self.labels_ph: labels}
        return feed_dict

    def save(self, name):
        self.saver.save(self.sess, name)

    def load(self, name):
        self.saver.restore(self.sess, name)

    def predict(self, batch, is_training=True):
        feed_dict = self.get_feed_dict(batch, self.dropout_rate if is_training else 0.)
        if is_training:
            out = self.sess.run([self.logits, self.loss, self.summary_op, self.global_step, self.train_op], feed_dict)
        else:
            # out = self.sess.run([self.predictions, self.loss, self.reg_loss], feed_dict)
            out = self.sess.run([self.predictions, self.loss], feed_dict)
        return out
