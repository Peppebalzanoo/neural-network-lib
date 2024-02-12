import dataset as ds
import network as net
import error as err


def run():
    training_set, test_set, training_labels, test_labels = ds.load_dataset()
    print("Trainig Set Shape: ", training_set.shape)
    print("Training Labels Shape: ", training_labels.shape)
    print("Test Set Shape: ", test_set.shape)
    print("Test Labels Shape: ", test_labels.shape)

    input_number_neurons = training_set.shape[0]
    output_number_neurons = training_labels.shape[0]
    mynet1 = net.create_network(input_number_neurons, [50], output_number_neurons)
    net.get_network_information(mynet1)

    idx_train, idx_val = ds.split_training_dataset(training_set, training_labels, 10)
    X_train = training_set[:, idx_train]
    Y_train = training_labels[:, idx_train]
    X_val = training_set[:, idx_val]
    Y_val = training_labels[:, idx_val]

    mynet2 = net.copy_network(mynet1)
    mynet3 = net.copy_network(mynet1)
    mynet4 = net.copy_network(mynet1)

    # net.backpropagation_train(mynet1, X_train, Y_train, X_val, Y_val, err.cross_entropy, 50, 0.00001)
    # net.resilient_train_rpropprof(mynet2, X_train, Y_train, X_val, Y_val, err.cross_entropy, 50, 0.00001)
    # net.resilient_train_rpropminus(mynet3, X_train, Y_train, X_val, Y_val, err.cross_entropy, 50, 0.00001)
    net.resilient_train_rpropminus_2(mynet4, X_train, Y_train, X_val, Y_val, err.cross_entropy, 50, 0.00001)
    # net.resilient_train_rpropplus(mynet5, X_train, Y_train, X_val, Y_val, err.cross_entropy, 50, 0.00001)


if __name__ == '__main__':
    run()
