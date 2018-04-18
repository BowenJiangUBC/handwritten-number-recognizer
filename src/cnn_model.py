#
# Created by bowenjiang on 4/9/18.
#
import tensorflow as tf



class cnn_single:
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return initial

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return initial

    def conv2d(self, x, shape, name="conv"):
        with tf.name_scope(name):
            w = tf.Variable(self.weight_variable(shape))
            b = tf.Variable(self.bias_variable([shape[3]]))
            return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1],
                                padding="SAME") + b

    def relu(self, x, name="relu"):
        with tf.name_scope(name):
            return tf.nn.relu(x)

    def max_pooling2x2(self, x, name="max_pooling"):
        with tf.name_scope(name):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding="SAME")

    def dense(self, x, size_in, size_out, name="dense"):
        with tf.name_scope(name):
            w = tf.Variable(self.weight_variable([size_in, size_out]))
            b = tf.Variable(self.bias_variable([size_out]))
            return tf.matmul(x, w) + b

    def dropout(self, x, prob, name="drouput"):
        with tf.name_scope(name):
            return tf.nn.dropout(x, prob)

    def get_name(self):
        return "cnn_single"

    def build_layers(self, input_images, keep_prob):
        with tf.name_scope("inference"):
            # divide 784 pixels into 28x28 array
            x = tf.reshape(input_images, [-1, 28, 28, 1])
            conv_1 = self.conv2d(x, [5, 5, 1, 32])
            relu_1 = self.relu(conv_1)
            max_pool_1 = self.max_pooling2x2(relu_1)

            conv_2 = self.conv2d(max_pool_1, [3, 3, 32, 16])
            relu_2 = self.relu(conv_2)
            max_pool_2 = self.max_pooling2x2(relu_2)

            conv_3 = self.conv2d(max_pool_2, [2, 2, 16, 8])
            relu_3 = self.relu(conv_3)
            max_pool_3 = self.max_pooling2x2(relu_3)

            max_pool_3_flat = tf.reshape(max_pool_3, [-1, 4 * 4 * 8])
            dense_1 = self.dense(max_pool_3_flat, 4 * 4 * 8, 64)
            dense_2 = self.dense(dense_1, 64, 32)
            dense_2_drop = self.dropout(dense_2, keep_prob)
            logits = self.dense(dense_2_drop, 32, 10)

            return logits

    def loss(self, logits, input_labels):
        with tf.name_scope("loss"):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=input_labels, logits=logits), name="cross_entropy_mean")
            tf.summary.scalar("loss", cross_entropy)
            return cross_entropy

    def train(self, loss, learning_rate):
        with tf.name_scope("training"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            return optimizer.minimize(loss)

    def evaluator(self, logits, input_labels):
        with tf.name_scope("evaluate"):
            correct_prediction = tf.equal(tf.argmax(logits, 1),
                                          tf.argmax(input_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return accuracy
