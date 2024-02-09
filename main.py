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
    mynet = net.create_network(input_number_neurons, [25, 50], output_number_neurons)
    net.get_network_information(mynet)

    # idx_train, idx_val = ds.split_training_dataset(training_set, training_labels)
    # print(type(idx_train), type(idx_val))
    # X_train = training_set[:, idx_train]
    # X_val = training_set[:, idx_val]
    # print("X_train.shape: ", X_train.shape, "X_val.shape: ", X_val.shape)

    # Z = net.forward_propagation(mynet, test_set)  # print("Accuray Network: ", net.get_accuracy_network(Z, test_labels))  #  # net.training_net(mynet, training_set, training_labels, err.cross_entropy, 10, 0.5)


if __name__ == '__main__':
    run()
