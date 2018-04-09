#
# Created by bowenjiang on 4/4/18.
#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

keep_prob = 1.0     # dropout rate
learning_rate = 0.0001      # learning rate
epoch_batch_size = 50


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return initial


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return initial


def conv2d(x, shape, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(weight_variable(shape))
        b = tf.Variable(bias_variable([shape[3]]))
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1],
                            padding="SAME") + b


def relu(x, name="relu"):
    with tf.name_scope(name):
        return tf.nn.relu(x)


def max_pooling2x2(x, name="max_pooling"):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding="SAME")


def dense(x, size_in, size_out, name="dense"):
    with tf.name_scope(name):
        w = tf.Variable(weight_variable([size_in, size_out]))
        b = tf.Variable(bias_variable([size_out]))
        return tf.matmul(x, w) + b


def dropout(x, prob, name="drouput"):
    with tf.name_scope(name):
        return tf.nn.dropout(x, prob)


def get_name():
    return "CNN_Adam"


def build_layers(input_images):
    with tf.name_scope("inference"):
        # divide 784 pixels into 28x28 array
        x = tf.reshape(input_images, [-1, 28, 28, 1])
        conv_1 = conv2d(x, [5, 5, 1, 32])
        relu_1 = relu(conv_1)
        max_pool_1 = max_pooling2x2(relu_1)

        conv_2 = conv2d(max_pool_1, [3, 3, 32, 16])
        relu_2 = relu(conv_2)
        max_pool_2 = max_pooling2x2(relu_2)

        conv_3 = conv2d(max_pool_2, [2, 2, 16, 8])
        relu_3 = relu(conv_3)
        max_pool_3 = max_pooling2x2(relu_3)

        max_pool_2_flat = tf.reshape(max_pool_3, [-1, 4*4*8])
        dense_1 = dense(max_pool_2_flat, 4*4*8, 64)
        dense_2 = dense(dense_1, 64, 32)
        dense_2_drop = dropout(dense_2, keep_prob)
        logits = dense(dense_2_drop, 32, 10)

        return logits


def loss(logits, input_labels):
    with tf.name_scope("loss"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=input_labels, logits=logits), name="cross_entropy_mean")
        tf.summary.scalar("loss", cross_entropy)
        return cross_entropy


def train(loss, learning_rate):
    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(loss)


def evaluator(logits, input_labels):
    with tf.name_scope("evaluate"):
        correct_prediction = tf.equal(tf.argmax(logits, 1),
                                      tf.argmax(input_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy


if __name__ == '__main__':

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    with tf.Graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, 784], name="input_images")
        input_labels = tf.placeholder(tf.float32, shape=[None, 10], name="input_labels")
        logits = build_layers(input_images)
        loss = loss(logits, input_labels)
        train = train(loss, learning_rate)
        evaluator = evaluator(logits, input_labels)

        # Initialization
        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)

        # Visualize graph
        writer = tf.summary.FileWriter("visualizations/ADAM_3_3/" + get_name())
        writer.add_graph(sess.graph)

        # Summaries
        merged_summary = tf.summary.merge_all()

        # Training
        for step in range(20000+1):
            image_batch, label_batch = mnist.train.next_batch(epoch_batch_size)
            loss_value, summary, _ = sess.run([loss, merged_summary, train], feed_dict={
                input_images: image_batch, input_labels: label_batch})
            writer.add_summary(summary, step)
            if step % 100 == 0:
                train_accuracy = sess.run(evaluator, feed_dict={
                    input_images: image_batch, input_labels: label_batch
                })
                summary = tf.Summary()
                summary.value.add(tag='Train Accuracy', simple_value=train_accuracy)
                writer.add_summary(summary, step)
                print("step %d, training accuracy %g" % (step, train_accuracy))
                validation_accuracy = sess.run(evaluator, feed_dict={
                    input_images: mnist.validation.images, input_labels: mnist.validation.labels
                })
                summary = tf.Summary()
                summary.value.add(tag='Validation Accuracy', simple_value=validation_accuracy)
                writer.add_summary(summary, step)
                print("step %d, validation accuracy %g" % (step, validation_accuracy))
                test_accuracy = sess.run(evaluator, feed_dict={
                    input_images: mnist.test.images, input_labels: mnist.test.labels
                })
                summary = tf.Summary()
                summary.value.add(tag='Test Accuracy', simple_value=test_accuracy)
                writer.add_summary(summary, step)
                print("step %d, test accuracy %g" % (step, test_accuracy))













