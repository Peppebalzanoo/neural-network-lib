import numpy
import numpy as np


def load_dataset():
    # Loat Dataset
    training_set = np.loadtxt('./MNIST_DATASET/mnist_train.csv', delimiter=",")
    test_set = np.loadtxt('./MNIST_DATASET/mnist_test.csv', delimiter=",")

    # Normalization Data
    training_set_normalized = (training_set[:, 1:]) / 255
    test_set_normalized = (test_set[:, 1:]) / 255

    # Extraction Lables: first column
    training_labels = training_set[:, 0]
    test_labels = test_set[:, 0]

    # Encoding Labels in One-Hot
    training_labels_one_hot = my_one_hot_encoding(training_labels)
    test_labels_one_hot = my_one_hot_encoding(test_labels)

    return (training_set_normalized.transpose(), test_set_normalized.transpose(), training_labels_one_hot.transpose(), test_labels_one_hot.transpose())


def my_one_hot_encoding(Y_col):
    count = len(Y_col)  # number of array
    size = 10  # size of each array

    # A list of count array with size = 10
    list_of_one_hot_labels = np.zeros((count, size), dtype=int)

    for i in range(0, count):
        curr_class = int(Y_col[i])
        curr_one_hot = np.zeros(10)
        curr_one_hot[curr_class] = 1
        list_of_one_hot_labels[i] = curr_one_hot

    return list_of_one_hot_labels


def split_training_dataset(X_train, Y_train):
    print("X_train.shape: ", X_train.shape, " Y_train.shape: ", Y_train.shape)

    # list_index_partitions = []
    # # i : 0 --> 10
    # for i in range(0, Y_train.shape[0]):
    #     partition = []
    #     # j : 0 --> 60.000
    #     for j in range(0, X_train.shape[1]):
    #         if Y_train[i, j] == 1:
    #             partition.append(j)
    #     list_index_partitions.append(partition)
    #
    # # Partition in k partitions
    # X_train_partiton = np.empty(10, dtype=object)
    # for partition in list_index_partitions:
    #     for i in range(0, len(partition)):
    #         X_train_partiton[i % 10] = np.append(X_train_partiton[i % 10], partition[i])
    #
    # # Shuffle of the k partitons
    # for i in range(0, 10):
    #     np.random.shuffle(X_train_partiton[i])
    #
    # # Create list of index
    # X_train_indexes = np.empty((), dtype=int)
    # for i in range(0, 9):
    #     X_train_indexes = np.append(X_train_indexes, X_train_partiton[i])
    #
    # X_val_indexes = np.array((), dtype=int)
    # X_val_indexes = np.append(X_val_indexes, X_train_partiton[9])
    #
    # return X_train_indexes, X_val_indexes


