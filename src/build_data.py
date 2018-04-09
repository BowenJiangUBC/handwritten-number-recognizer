#
# Created by bowenjiang on 4/9/18.
#

import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import pickle


'''
Create dataset of digit sequnces by concatenating mnist digits

Params:
    data - dataset to use (train/validation/test)
    dataset_size - size of dataset to generate
    length - the length of digit sequence


'''


def build_data(data, dataset_size, length):

    images = []
    labels = []
    for i in range(dataset_size):
        s_indices = [random.randint(0, data.num_examples - 1) for p in range(length)]
        image = []
        label = []
        for j in range(length):
            img_tmp = np.reshape(data.images[s_indices[j]], [28, 28])
            # plt.imshow(image, cmap='gray')
            # plt.show()
            if len(image) == 0:
                image = img_tmp
                label = data.labels[s_indices[j]]
            else:
                image = np.append(image, img_tmp, axis=1)
                label = np.vstack([label, data.labels[s_indices[j]]])
        images.append(image)
        labels.append(label)
    return {"images": images, "labels": labels}


def load_data(file_name, debug = False):
    dataset = pickle.load(open(file_name, "rb"))
    if debug:
        print(dataset["labels"][0])
        plt.imshow(dataset["images"][0], cmap='gray')
        plt.show()
    return dataset

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    print("Creating training dataset")
    train_dataset = build_data(mnist.train, 130000, 5)
    pickle.dump(train_dataset, open("train.pkl", "wb"))
    print("Creating validation dataset")
    validation_dataset = build_data(mnist.validation, 15000, 5)
    pickle.dump(validation_dataset, open("validation.pkl", "wb"))
    print("Creating testing dataset")
    test_dataset = build_data(mnist.test, 30000, 5)
    pickle.dump(test_dataset, open("test.pkl", "wb"))
    print("Done")

    load_data("train.pkl", True)
    load_data("validation.pkl", True)
    load_data("test.pkl", True)
