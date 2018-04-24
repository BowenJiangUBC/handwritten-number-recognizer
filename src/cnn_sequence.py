#
# Created by bowenjiang on 4/9/18.
#

import tensorflow as tf
import numpy as np

class cnn_sequence:

    def conv2d(self, x, shape, name="conv"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal(shape))
            b = tf.Variable(tf.constant(0.1, shape=[shape[3]]))
            return tf.nn.conv2d(x, w,
                                strides=[1, 1, 1, 1], padding="SAME") + b

    def relu(self, x, name="relu"):
        with tf.name_scope(name):
            return tf.nn.relu(x)

    def max_pooling2x2(self, x, name="max_pooling"):
        with tf.name_scope(name):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding="SAME")

    def dense(self, x, size_in, size_out, name="dense"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([size_in, size_out]))
            b = tf.Variable(tf.constant(0.1, shape=[size_out]))
            return tf.matmul(x, w) + b

    def dropout(self, x, prob, name="dropout"):
        with tf.name_scope(name):
            return tf.nn.dropout(x, prob)

    def get_name(self):
        return "cnn_sequence"

    def build_layers(self, input_images, keep_prob):
        with tf.name_scope("layers"):
            x = tf.reshape(input_images, [-1, 64, 64, 1])
            conv_1 = self.conv2d(x, [5, 5, 1, 16])
            relu_1 = self.relu(conv_1)
            max_pool_1 = self.max_pooling2x2(relu_1)

            conv_2 = self.conv2d(max_pool_1, [5, 5, 16, 32])
            relu_2 = self.relu(conv_2)
            max_pool_2 = self.max_pooling2x2(relu_2)

            conv_3 = self.conv2d(max_pool_2, [5, 5, 32, 64])
            relu_3 = self.relu(conv_3)
            max_pool_3 = self.max_pooling2x2(relu_3)

            max_pool_3_flat = tf.reshape(max_pool_3, [-1, 8 * 8 * 64])
            max_pool_3_flat_drop = tf.nn.dropout(max_pool_3_flat, keep_prob)
            dense_1 = self.relu(self.dense(max_pool_3_flat_drop, 8 * 8 * 64, 1024))
            dense_1_drop = tf.nn.dropout(dense_1, keep_prob)
            logits = []
            for i in range(5):
                temp = self.dense(dense_1_drop, 1024, 10)
                logits.append(temp)
            return tf.stack(logits, axis=1)

    def loss(self, logits, input_labels):
        with tf.name_scope("loss"):
            cross_entroy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=input_labels, logits=logits), name="cross_entropy_mean")
            tf.summary.scalar("loss", cross_entroy)
            return cross_entroy

    def train(self, loss, learning_rate):
        with tf.name_scope("training"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            return optimizer.minimize(loss)

    def evaluator(self, logits, input_labels):
        with tf.name_scope("evaluate"):
            # convert our raw labels into matrix of nx5
            # while each row is the values of five digit
            logits = tf.argmax(logits, 2)
            labels = tf.argmax(input_labels, 2)
            diff = tf.subtract(labels, logits, name="sub")
            individual_corrects = tf.equal(diff, 0, name="individual_zero")
            individual_corrects = tf.reduce_mean(tf.cast(individual_corrects, tf.float32))
            sequence_corrects = tf.count_nonzero(diff, axis=1, name="count_nonzero")
            sequence_corrects = tf.equal(sequence_corrects, 0, name="is_zero")
            sequence_corrects = tf.reduce_mean(tf.cast(sequence_corrects, tf.float32))

            return individual_corrects, sequence_corrects


    def converter(self, input_labels):
        with tf.name_scope("convert"):
            return tf.argmax(input_labels, 2)




