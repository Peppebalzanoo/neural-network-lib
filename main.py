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
    mynet = net.create_network(input_number_neurons, [50], output_number_neurons)
    net.get_network_information(mynet)

    idx_train, idx_val = ds.split_training_dataset(training_set, training_labels, 10)
    X_train = training_set[:, idx_train]
    Y_train = training_labels[:, idx_train]
    X_val = training_set[:, idx_val]
    Y_val = training_labels[:, idx_val]

    net.resilient_train(mynet, X_train, Y_train, X_val, Y_val, err.cross_entropy, 300, 0.00001)


if __name__ == '__main__':
    run()
