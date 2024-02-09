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

    return training_set_normalized.transpose(), test_set_normalized.transpose(), training_labels_one_hot.transpose(), test_labels_one_hot.transpose()


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


def split_training_dataset(X_train, Y_train, k):
    X_fold = np.array_split(np.array([], dtype=int), k)
    # i : 0 to 10
    for i in range(0, Y_train.shape[0]):
        # partion for class i
        partition = np.array([], dtype=int)
        # j : 0  to 60.000
        for j in range(0, X_train.shape[1]):
            if Y_train[i, j] == 1:
                partition = np.append(partition, j)
        # split partion in k fold
        partition = np.array_split(partition, k)
        for idx in range(0, k):
            X_fold[idx] = np.append(X_fold[idx], partition[idx])
            # shuffle partion
            np.random.shuffle(X_fold[idx])

    return np.concatenate(X_fold[0:9]), np.concatenate(X_fold[9:])
