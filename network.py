import numpy as np
import activation as actfun
import copy as cp


def create_network(input_number, list_numbers_of_hidden, output_number):
    sigma = 0.001
    weights_list = []
    baiases_list = []
    activations_fun_list = []

    prev_connections_number = input_number
    for curr_hidden_number in list_numbers_of_hidden:
        # Generate baiases array for curr_number neurons (one baias for each neuron)
        baiases_list.append(sigma * np.random.normal(size=[curr_hidden_number, 1]))
        # Generate weights matrix for curr_number neurons
        weights_list.append(sigma * np.random.normal(size=[curr_hidden_number, prev_connections_number]))
        activations_fun_list.append(actfun.numpy_tanh)
        # update prev_conncetion_number
        prev_connections_number = curr_hidden_number

    # For the output layer
    baiases_list.append(sigma * np.random.normal(size=[output_number, 1]))
    weights_list.append(sigma * np.random.normal(size=[output_number, prev_connections_number]))
    activations_fun_list.append(actfun.identity)

    mynet = {"W": weights_list, "B": baiases_list, "ActFun": activations_fun_list, "Depth": len(weights_list)}

    return mynet


def copy_network(net):
    return cp.deepcopy(net)


def get_network_information(net):
    input_layer_number = output_layer_number = 1
    hidden_layers_numbers = net['Depth'] - 1
    print("\nDepth network: ", net["Depth"])
    print("Number of input layer: ", input_layer_number)
    print("Number of input neurons:", net["W"][0].shape[1])
    print("Number of hidden layers: ", hidden_layers_numbers)
    print("Number of hidden neurons: ", [net["W"][i].shape[0] for i in range(0, hidden_layers_numbers)])
    print("Number of output layer: ", output_layer_number)
    print("Number of output neurons:", net["W"][(net["Depth"] - 1)].shape[0])
    print("Weights shape: ", [net["W"][i].shape for i in range(0, (input_layer_number + hidden_layers_numbers + output_layer_number) - 1)])
    print("Activation shape: ", [(net["ActFun"][i]).__name__ for i in range(0, net["Depth"])])


def get_accuracy_network(prediction_labels_set, gold_labels_set):
    total_cases = gold_labels_set.shape[1]
    good_cases = 0
    for i in range(0, gold_labels_set.shape[1]):
        gold_label = np.argmax(gold_labels_set[:, i])
        my_prediction_label = np.argmax(prediction_labels_set[:, i])
        if gold_label == my_prediction_label:
            good_cases += 1
    return good_cases / total_cases


def forward_propagation(net, train_set):
    depth = net["Depth"]
    Z_layer = train_set
    for layer in range(0, depth):
        W = net["W"][layer]
        B = net["B"][layer]
        A_layer = np.matmul(W, Z_layer) + B
        g = net["ActFun"][layer]
        # update Z: g(A_layer)
        Z_layer = g(A_layer)
    return Z_layer


def forward_propagation_training(net, train_set):
    depth = net["Depth"]

    Z_derived_list = []
    Z_list = [train_set]

    for layer in range(0, depth):
        W = net["W"][layer]
        B = net["B"][layer]
        g = net["ActFun"][layer]

        A_layer = np.matmul(W, Z_list[layer]) + B
        # Append and update Z
        Z_list.append(g(A_layer))
        Z_derived_list.append(g(A_layer, 1))

    return Z_list, Z_derived_list


def backpropagation(net, train_set, gold_labels_set, error_function):
    # * STEP 1 : FORWORD STEP
    Z_list, Z_derived_list = forward_propagation_training(net, train_set)

    # print("len(z_list): ", len(Z_list))
    # print("len(Z_derived_list): ", len(Z_derived_list))
    # print("len(net[W]): ", len(net["W"]))

    # * STEP_2: COMPUTE DELTA VALUES AND BACK PROPAGATE THEM
    delta_values_list = []

    for i in range(net["Depth"], 0, -1):
        if i == net["Depth"]:
            # Compute delta-k for k-neurons of output
            der_error_function = error_function(Z_list[i], gold_labels_set, 1)
            delta_k = der_error_function * Z_derived_list[i - 1]
            delta_values_list.insert(0, delta_k)

        else:
            # Compute delta-h for h-neurons of output
            W = net["W"][i]
            # print("W.shape", W.shape)
            # print("delta[0].shape", delta_values_list[0].shape)
            delta_h = (np.matmul(W.transpose(), delta_values_list[0])) * Z_derived_list[i - 1]
            delta_values_list.insert(0, delta_h)

    # * STEP 3: COMPUTE ALL PARTIAL DERIVATE
    der_partial_list = []
    for i in range(0, net["Depth"]):
        local_row = np.matmul(delta_values_list[i], Z_list[i].transpose())
        der_partial_list.append(local_row)

    return der_partial_list


def back_propagation_training(net, train_set, train_labels, error_fun, epoche_number=0, eta=0.1):
    Z_train = forward_propagation(net, train_set)
    err_train = error_fun(Z_train, train_labels)

    # Z_test = forward_propagation(net, val_set)
    # error_val = error_fun(Z_test, val_labels)

    print("Epoca: ", 0, "Train error: ", err_train, "Accuracy Train: ", get_accuracy_network(Z_train, train_labels))
    epoca = 0
    while epoca < epoche_number:
        der_partial_list = backpropagation(net, train_set, train_labels, error_fun)
        # Gradient Descent: Update Weights
        for layer in range(0, net["Depth"]):
            net["W"][layer] = net["W"][layer] - (eta * der_partial_list[layer])

        Z_train = forward_propagation(net, train_set)
        err_train = error_fun(Z_train, train_labels)
        print("Epoca: ", epoca, "Train error: ", err_train, "Accuracy Train: ", get_accuracy_network(Z_train, train_labels))

        epoca += 1
