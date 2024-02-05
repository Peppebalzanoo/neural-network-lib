import numpy as np

def load_dataset():
    # Loat Dataset
    training_set = np.loadtxt('./MNIST_DATASET/mnist_train.csv', delimiter=",")
    test_set = np.loadtxt('./MNIST_DATASET/mnist_test.csv', delimiter=",")

    # Normalization Data
    training_set_normalized = training_set[:, 1:] * (1/255)
    test_set_normalized = test_set[:, 1:] * (1/255)

    # Extraction Lables
    training_labels = training_set[:, 0]
    test_labels = test_set[:, 0]

    # Encoding Labels in One-Hot
    training_labels_one_hot = my_one_hot_encoding(training_labels)
    test_labels_one_hot = my_one_hot_encoding(test_labels)

    return (training_set_normalized.transpose(),
            test_set_normalized.transpose(),
            training_labels_one_hot.transpose(),
            test_labels_one_hot.transpose())




def my_one_hot_encoding(labels_set):
    count = len(labels_set)  # number of array
    size = 10  # size of each array

    # A list of count array with size = 10
    list_of_one_hot_labels = np.zeros((count, size), dtype=int)

    for i in range(0, count):
        curr_class = int(labels_set[i])
        # Array with size = 10
        curr_one_hot = np.zeros(10)
        # Array with size = 10 and one element = 1
        curr_one_hot[curr_class] = 1
        list_of_one_hot_labels[i] = curr_one_hot
    return list_of_one_hot_labels
