#
# Created by bowenjiang on 4/9/18.
#

import tensorflow as tf
from src.cnn_sequence import cnn_sequence
from src.build_data import load_data
import numpy as np
import time

keep_prob = 0.5
learning_rate = 0.001
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

def evaluate(sess, inference, converter, images, labels, batch_size=128):
    assert len(images) == len(labels), "Images and labels length mismatch"
    nrof_images = len(images)
    nrof_batches = nrof_images // batch_size + 1

    ind_pred = None
    seq_pred = None

    for i in range(nrof_batches):
        start = i*batch_size
        end = nrof_images if start+batch_size > nrof_images else start+batch_size
        image_batch = np.asarray(images[start:end])
        labels_batch = np.asarray(labels[start:end])

        logits = sess.run(inference, feed_dict={
            input_images: image_batch
        })
        predicts = tf.argmax(logits, 2)
        true_labels = sess.run(converter, feed_dict={
            input_labels: labels_batch
            })

        diff = tf.subtract(true_labels, predicts, name="sub")
        individual_corrects = tf.equal(diff, 0, name="individual_zero")
        # individual_corrects = tf.reduce_mean(tf.cast(individual_corrects, tf.float32))
        sequence_corrects = tf.count_nonzero(diff, axis=1, name="count_nonzero")
        sequence_corrects = tf.equal(sequence_corrects, 0, name="is_zero")
        # sequence_corrects = tf.reduce_mean(tf.cast(sequence_corrects, tf.float32))

        if ind_pred is not None:
            ind_pred = tf.concat([ind_pred, individual_corrects], 0)
        else:
            ind_pred = individual_corrects

        if seq_pred is not None:
            seq_pred = tf.concat([seq_pred, sequence_corrects], 0)
        else:
            seq_pred = sequence_corrects

    ind_acc = tf.reduce_mean(tf.cast(ind_pred, tf.float32))
    seq_acc = tf.reduce_mean(tf.cast(seq_pred, tf.float32))

    return ind_acc.eval(), seq_acc.eval()


if __name__ == '__main__':
    train_data = load_data("train.p", False)
    validation_data = load_data("validation.p", False)
    test_data = load_data("test.p", False)

    with tf.Graph().as_default():
        cnn = cnn_sequence()

        input_images = tf.placeholder(tf.float32, shape=[None, 64, 64], name="input_images")
        input_labels = tf.placeholder(tf.float32, shape=[None, 5, 10], name="input_labels")
        logits = cnn.build_layers(input_images, keep_prob)
        loss = cnn.loss(logits, input_labels)
        train = cnn.train(loss, learning_rate)
        evaluator = cnn.evaluator(logits, input_labels)
        inference = cnn.build_layers(input_images, keep_prob=1.0)
        converter = cnn.converter(input_labels)

        # Initialization
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        init = tf.global_variables_initializer()
        sess.run(init)

        # Visualize graph
        writer = tf.summary.FileWriter("visualizations/Sequence_cnn/" + cnn.get_name())
        writer.add_graph(sess.graph)

        # Summaries
        merged_summary = tf.summary.merge_all()

        for step in range(10000+1):
            start = time.time()
            image_batch, label_batch = next_batch(train_data, epoch_batch_size)
            loss_value, summary, _ = sess.run([loss, merged_summary, train], feed_dict={
                input_images: image_batch, input_labels: label_batch})
            writer.add_summary(summary, step)
            if step % 100 == 0:
                train_ind_accuracy, train_seq_accuracy = sess.run(evaluator, feed_dict={
                    input_images: image_batch, input_labels: label_batch
                })

                summary = tf.Summary()
                summary.value.add(tag='Train individual Accuracy', simple_value=train_ind_accuracy)
                summary.value.add(tag='Train sequence Accuracy', simple_value=train_seq_accuracy)
                writer.add_summary(summary, step)
                print("step %d, training individual accuracy %g" % (step, train_ind_accuracy))
                print("step %d, training sequence accuracy %g" % (step, train_seq_accuracy))

                val_ind_accuracy, val_seq_accuracy = evaluate(sess, inference, converter, validation_data["images"],
                                                              validation_data["labels"])

                summary = tf.Summary()
                summary.value.add(tag='Validation individual Accuracy', simple_value=val_ind_accuracy)
                summary.value.add(tag='Validation sequence Accuracy', simple_value=val_seq_accuracy)
                writer.add_summary(summary)
                print("validation individual accuracy %g" % val_ind_accuracy)
                print("validation sequence accuracy %g" % val_seq_accuracy)

                test_ind_accuracy, test_seq_accuracy = evaluate(sess, inference, converter, test_data["images"],
                                                                test_data["labels"])

                summary = tf.Summary()
                summary.value.add(tag='Test individual Accuracy', simple_value=test_ind_accuracy)
                summary.value.add(tag='Test sequence Accuracy', simple_value=test_seq_accuracy)
                writer.add_summary(summary)
                print("test individual accuracy %g" % test_ind_accuracy)
                print("test sequence accuracy %g" % test_seq_accuracy)

                print("Time took for 100 steps %d: %.3f" % (step, time.time() - start))
                start = time.time()




