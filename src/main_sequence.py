#
# Created by bowenjiang on 4/9/18.
#

import tensorflow as tf
from src.cnn_sequence import cnn_sequence
from src.build_data import load_data
import numpy as np

keep_prob = 1.0
learning_rate = 0.0001
epoch_batch_size = 150


def next_batch(data, num):
    '''
    Return a total of `num` random samples and labels.
    '''

    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    images = [data["images"][i] for i in idx]
    labels = [data["labels"][i] for i in idx]

    return np.asarray(images), np.asarray(labels)


if __name__ == '__main__':
    train_data = load_data("train.p", False)
    validation_data = load_data("validation.p", False)
    test_data = load_data("test.p", False)

    with tf.Graph().as_default():
        cnn = cnn_sequence()

        input_images = tf.placeholder(tf.float32, shape=[None, 28, 140], name="input_images")
        input_labels = tf.placeholder(tf.float32, shape=[None, 5, 10], name="input_labels")
        logits = cnn.build_layers(input_images, keep_prob)
        loss = cnn.loss(logits, input_labels)
        train = cnn.train(loss, learning_rate)

        # Initialization
        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)

        # Visualize graph
        writer = tf.summary.FileWriter("visualizations/Sequence_cnn/" + cnn.get_name())
        writer.add_graph(sess.graph)

        # Summaries
        merged_summary = tf.summary.merge_all()

        for step in range(10000+1):
            image_batch, label_batch = next_batch(train_data, epoch_batch_size)
            loss_value, summary, _ = sess.run([loss, merged_summary, train], feed_dict={
                input_images: image_batch, input_labels: label_batch})
            writer.add_summary(summary, step)
            evaluator = cnn.evaluator(logits, input_labels)
            if step % 100 == 0:
                train_ind_accuracy, train_seq_accuracy, batch_size = sess.run(evaluator, feed_dict={
                    input_images: image_batch, input_labels: label_batch
                })
                summary = tf.Summary()
                summary.value.add(tag='Train individual Accuracy', simple_value=train_ind_accuracy)
                summary.value.add(tag='Train sequence Accuracy', simple_value=train_seq_accuracy)
                writer.add_summary(summary, step)
                print("step %d, training individual accuracy %g" % (step, train_ind_accuracy))
                print("step %d, training sequence accuracy %g" % (step, train_seq_accuracy))

                val_ind_accuracy, val_seq_accuracy, batch_size = sess.run(evaluator, feed_dict={
                    input_images: validation_data["images"], input_labels: validation_data["labels"]
                })
                summary = tf.Summary()
                summary.value.add(tag='Validation individual Accuracy', simple_value=val_ind_accuracy)
                summary.value.add(tag='Validation sequence Accuracy', simple_value=val_seq_accuracy)
                writer.add_summary(summary, step)
                print("step %d, validation individual accuracy %g" % (step, val_ind_accuracy))
                print("step %d, validation sequence accuracy %g" % (step, val_seq_accuracy))

                test_ind_accuracy, test_seq_accuracy, batch_size = sess.run(evaluator, feed_dict={
                    input_images: test_data["images"], input_labels: test_data["labels"]
                })
                summary = tf.Summary()
                summary.value.add(tag='Test individual Accuracy', simple_value=test_ind_accuracy)
                summary.value.add(tag='Test sequence Accuracy', simple_value=test_seq_accuracy)
                writer.add_summary(summary, step)
                print("step %d, test individual accuracy %g" % (step, test_ind_accuracy))
                print("step %d, test sequence accuracy %g" % (step, test_seq_accuracy))
