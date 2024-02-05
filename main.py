import dataset as ds
import network as net

def run():
    training_set, test_set, training_labels, test_labels = ds.load_dataset()
    print("Trainig Set Shape: ", training_set.shape)
    print("Training Labels Shape: ", training_labels.shape)
    print("Test Set Shape: ", test_set.shape)
    print("Test Labels Shape: ", test_labels.shape)

    input_number_neurons = training_set.shape[0]
    output_number_neurons = training_labels.shape[0]
    mynet = net.create_new_network(input_number_neurons, [25, 50], output_number_neurons)
    net.get_network_information(mynet)

if __name__ == '__main__':
    run()
