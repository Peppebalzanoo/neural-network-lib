import numpy as np
import activation as actfun

def create_new_network(input_number, list_numbers_of_hidden, output_number):
    sigma = 0.001
    weights_list = []
    baiases_list = []
    activations_fun_list = []

    prev_number = input_number
    for curr_number in list_numbers_of_hidden:
        # Generate curr_number baias for curr_number neurons
        baiases_list.append(sigma * np.random.normal(size=[curr_number, 1]))
        # Generate curr_number weights for curr_number neurons
        weights_list.append(sigma * np.random.normal(size=[curr_number, prev_number]))
        activations_fun_list.append(actfun.numpy_tanh)

    # For the output layer
    # Generate number_output baias for number_output neurons
    baiases_list.append(sigma * np.random.normal(size=[output_number, 1]))
    # Generate number_output weights for number_output neurons
    weights_list.append(sigma * np.random.normal(size=[output_number, prev_number]))
    activations_fun_list.append(actfun.identity)

    mynet = {'W': weights_list, 'B': baiases_list, 'ActFun': activations_fun_list, 'Depth': len(weights_list)}

    return mynet


def get_network_information(net):
    input_layer_number = output_layer_number = 1
    hidden_layers_numbers = net['Depth'] - 1

    print("\nNumber of input layer: ", input_layer_number)
    print("Number of input neurons:", net["W"][0].shape[1])
    print("Number of hidden layers: ", hidden_layers_numbers)
    print("Number of hidden neurons: ", [net["W"][i].shape[0] for i in range(0, hidden_layers_numbers)])
    print("Number of output layer: ", output_layer_number)
    print("Number of output neurons:", net["W"][(net["Depth"]-1)].shape[0])
    print("Weights shape: ", [net["W"][i].shape for i in range(0, (input_layer_number+hidden_layers_numbers+output_layer_number)-1)])
    print("Activation shape: ", [(net["ActFun"][i]).__name__ for i in range(0, net["Depth"])])
