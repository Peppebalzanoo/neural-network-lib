import activation
import dataset as ds
import network as net
import error as err


def run():
    training_set, test_set, training_labels, test_labels = ds.load_dataset()
    print("Trainig Set Shape: ", training_set.shape)
    print("Training Labels Shape: ", training_labels.shape)
    print("Test Set Shape: ", test_set.shape)
    print("Test Labels Shape: ", test_labels.shape)

    idx_train, idx_val = ds.split_training_dataset(training_set, training_labels, 10)
    X_train = training_set[:, idx_train]
    Y_train = training_labels[:, idx_train]
    X_val = training_set[:, idx_val]
    Y_val = training_labels[:, idx_val]

    input_number_neurons = training_set.shape[0]
    output_number_neurons = training_labels.shape[0]
    mynet1 = net.create_network(input_number_neurons,
                                [25],
                                output_number_neurons,
                                0.001,
                                activation.tanh,
                                [activation.tanh])
    net.get_network_information(mynet1)

    mynet2 = net.copy_network(mynet1)

    eta_minus = 0.5
    eta_plus = 1.2
    delta_zero = 0.0125
    delta_min = 0.00001
    delta_max = 1

    net.resilient_train_rpropminus(mynet1,
                                   X_train, Y_train, X_val, Y_val,
                                   err.cross_entropy,
                                   50,
                                   eta_minus, eta_plus,
                                   delta_zero, delta_min, delta_max)

    net.resilient_train_rpropplus(mynet2,
                                  X_train, Y_train, X_val, Y_val,
                                  err.cross_entropy,
                                  50,
                                  eta_minus, eta_plus,
                                  delta_zero, delta_min, delta_max)


if __name__ == '__main__':
    run()
