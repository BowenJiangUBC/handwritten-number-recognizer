#
# Created by bowenjiang on 4/4/18.
#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from src.cnn_model import cnn_single

keep_prob = 1.0     # dropout rate
learning_rate = 0.0001      # learning rate
epoch_batch_size = 100

if __name__ == '__main__':

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    with tf.Graph().as_default():

        cnn = cnn_single()
        input_images = tf.placeholder(tf.float32, shape=[None, 784], name="input_images")
        input_labels = tf.placeholder(tf.float32, shape=[None, 10], name="input_labels")
        logits = cnn.build_layers(input_images, keep_prob)
        loss = cnn.loss(logits, input_labels)
        train = cnn.train(loss, learning_rate)
        evaluator = cnn.evaluator(logits, input_labels)

        # Initialization
        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)

        # Visualize graph
        writer = tf.summary.FileWriter("visualizations/ADAM_3_3/" + cnn.get_name())
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













