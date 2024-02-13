import numpy as np

import activation
import activation as actfun
import copy as cp


def create_network(input_number, list_numbers_of_hidden, output_number, sigma=0.001, activation_defualt=activation.tanh, list_function=None):
    weights_list = []
    baiases_list = []

    if np.isscalar(list_numbers_of_hidden):
        list_numbers_of_hidden = [list_numbers_of_hidden]

    prev_connections_number = input_number
    for curr_hidden_number in list_numbers_of_hidden:
        # Generate baiases array for curr_number neurons (one baias for each neuron)
        baiases_list.append(sigma * np.random.normal(size=[curr_hidden_number, 1]))
        # Generate weights matrix for curr_number neurons
        weights_list.append(sigma * np.random.normal(size=[curr_hidden_number, prev_connections_number]))
        # update prev_conncetion_number
        prev_connections_number = curr_hidden_number
    activations_fun_list = set_activation_functions(len(weights_list), activation_defualt, list_function)

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


def set_activation_functions(net_depth, activation_default, list_functions):
    # Set activation defaul for all layer
    if list_functions is None or len(list_functions) == 0:
        return [activation_default for _ in range(0, net_depth)]
    # Set one activation function for all layers without last one
    elif len(list_functions) == 1:
        return [list_functions[0] for _ in range(0, net_depth)]
    # Set specific activation for layers and activation defualt for all others layers without last one
    elif len(list_functions) < net_depth:
        ret1 = [list_functions[layer] for layer in range(0, len(list_functions))]
        ret2 = [activation_default for _ in range(len(list_functions), net_depth)]
        return ret1 + ret2
    else:
        raise Exception("Illegal number of arguments: set_activation_function")


def get_accuracy_network(Z_out, Y_gold):
    total_cases = Y_gold.shape[1]
    good_cases = 0
    for i in range(0, Y_gold.shape[1]):
        gold_label = np.argmax(Y_gold[:, i])
        my_prediction_label = np.argmax(Z_out[:, i])
        if gold_label == my_prediction_label:
            good_cases += 1
    return good_cases / total_cases


def forward_propagation(net, X_train):
    depth = net["Depth"]
    Z_layer = X_train
    for layer in range(0, depth):
        W = net["W"][layer]
        B = net["B"][layer]
        # Computation of layer
        A_layer = np.matmul(W, Z_layer) + B
        g = net["ActFun"][layer]
        # Output of layer: g(A_layer)
        Z_layer = g(A_layer)
    return Z_layer


def forward_propagation_training(net, X_train):
    depth = net["Depth"]

    Z_derived_list = []
    # Initialize Z_list with input
    Z_list = [X_train]

    for layer in range(0, depth):
        W = net["W"][layer]
        B = net["B"][layer]
        g = net["ActFun"][layer]
        # Computation of layer
        A_layer = np.matmul(W, Z_list[layer]) + B
        # Output of layer: g(A_layer) and append
        Z_list.append(g(A_layer))
        Z_derived_list.append(g(A_layer, 1))

    return Z_list, Z_derived_list


def gradient_descent_training(net, der_partial_list, eta):
    for layer in range(0, net["Depth"]):
        net["W"][layer] = net["W"][layer] - (eta * der_partial_list[layer])


def backpropagation(net, X_train, Y_gold, error_function):
    # * STEP 1 : FORWARD-STEP * #
    # Z_list have depth + 1 elements (include Z_input)
    # Z_derived_list have depth elements
    Z_list, Z_derived_list = forward_propagation_training(net, X_train)

    # * STEP 2: COMPUTE DELTA-VALUES AND BACK-PROPAGATE THEM * #
    delta_values_list = []
    for layer in range(net["Depth"], 0, -1):
        if layer == net["Depth"]:
            # Compute delta-k for k-neurons of output
            der_error_function = error_function(Z_list[layer], Y_gold, 1)
            delta_k = der_error_function * Z_derived_list[layer - 1]
            delta_values_list.insert(0, delta_k)
        else:
            # Compute delta-h for h-neurons of output
            W = net["W"][layer]
            # print("W.shape", W.shape)
            # print("delta[0].shape", delta_values_list[0].shape)
            delta_h = (np.matmul(W.transpose(), delta_values_list[0])) * Z_derived_list[layer - 1]
            delta_values_list.insert(0, delta_h)

    # * STEP 3: COMPUTE ALL PARTIAL DERIVATE (LOCAL ROW) * #
    der_partial_list = []
    for layer in range(0, net["Depth"]):
        local_row = np.matmul(delta_values_list[layer], Z_list[layer].transpose())
        der_partial_list.append(local_row)

    return der_partial_list


def backpropagation_train(net, X_train, Y_train, X_val, Y_val, error_function, epoche_number=0, eta=0.1):
    Z_train = forward_propagation(net, X_train)
    err_train = error_function(Z_train, Y_train)
    Z_val = forward_propagation(net, X_val)
    error_val = error_function(Z_val, Y_val)
    print("Epoca: ", -1, "Train error: ", err_train, "Accuracy Train: ", get_accuracy_network(Z_train, Y_train), "Validation error: ", error_val, "Accuracy Validation: ",
          get_accuracy_network(Z_val, Y_val))

    epoca = 0
    while epoca < epoche_number:
        der_partial_list = backpropagation(net, X_train, Y_train, error_function)

        # * STEP 4: GRADIENT DESCENT (UPDATE WEIGHTS) * #
        gradient_descent_training(net, der_partial_list, eta)

        Z_train = forward_propagation(net, X_train)
        err_train = error_function(Z_train, Y_train)

        Z_val = forward_propagation(net, X_val)
        error_val = error_function(Z_val, Y_val)

        if epoca == epoche_number - 1:
            print("Epoca: ", epoca, "Train error: ", err_train, "Accuracy Train: ", get_accuracy_network(Z_train, Y_train), "Validation error: ", error_val, "Accuracy Validation: ",
                  get_accuracy_network(Z_val, Y_val))

        epoca += 1


def resilient_train_rpropminus(net, X_train, Y_train, X_val, Y_val, error_function, epoche_number=0, eta_minus=0.5, eta_plus=1.2, delta_zero=0.0125, delta_min=0.00001, delta_max=1):
    Z_train = forward_propagation(net, X_train)
    err_train = error_function(Z_train, Y_train)
    Z_val = forward_propagation(net, X_val)
    error_val = error_function(Z_val, Y_val)
    print("Epoca: ", -1, "Train error: ", err_train, "Accuracy Train: ", get_accuracy_network(Z_train, Y_train), "Validation error: ", error_val, "Accuracy Validation: ",
          get_accuracy_network(Z_val, Y_val))

    der_list = []
    delta_ij = []
    for e in range(0, epoche_number):
        delta_ij.append([delta_zero] * net["Depth"])

    epoca = 0
    while epoca < epoche_number:
        # append matrix of partial derivatives of layer for each epoc
        der_list.append(backpropagation(net, X_train, Y_train, error_function))

        for layer in range(0, net["Depth"]):
            if epoca > 1:
                # retrive the matrix of derivatives
                prev_derivatives = der_list[epoca - 1][layer]
                curr_derivatives = der_list[epoca][layer]
                prod_der = prev_derivatives * curr_derivatives

                # new list of matrix with element calculated
                delta_ij[epoca][layer] = np.where(prod_der > 0, np.minimum(delta_ij[epoca - 1][layer] * eta_plus, delta_max),
                                                  np.where(prod_der < 0, np.maximum(delta_ij[epoca - 1][layer] * eta_minus, delta_min), delta_ij[epoca - 1][layer]))

                net["W"][layer] = net["W"][layer] - (np.sign(der_list[epoca][layer]) * delta_ij[epoca][layer])

        Z_train = forward_propagation(net, X_train)
        err_train = error_function(Z_train, Y_train)

        Z_val = forward_propagation(net, X_val)
        error_val = error_function(Z_val, Y_val)

        print("Epoca: ", epoca, "Train error: ", err_train, "Accuracy Train: ", get_accuracy_network(Z_train, Y_train), "Validation error: ", error_val, "Accuracy Validation: ",
              get_accuracy_network(Z_val, Y_val))

        epoca += 1


def resilient_train_rpropplus(net, X_train, Y_train, X_val, Y_val, error_function, epoche_number=0, eta_minus=0.5, eta_plus=1.2, delta_zero=0.0125, delta_min=0.00001, delta_max=1):
    Z_train = forward_propagation(net, X_train)
    err_train = error_function(Z_train, Y_train)
    Z_val = forward_propagation(net, X_val)
    error_val = error_function(Z_val, Y_val)
    print("Epoca: ", -1, "Train error: ", err_train, "Accuracy Train: ", get_accuracy_network(Z_train, Y_train), "Validation error: ", error_val, "Accuracy Validation: ",
          get_accuracy_network(Z_val, Y_val))

    der_list = []

    delta_ij = []
    delta_wij = []
    for e in range(0, epoche_number):
        delta_ij.append([delta_zero] * net["Depth"])
        delta_wij.append([0] * net["Depth"])

    epoca = 0
    while epoca < epoche_number:
        der_list.append(backpropagation(net, X_train, Y_train, error_function))
        for layer in range(0, net["Depth"]):
            if epoca == 0:
                delta_wij[epoca][layer] = np.where(der_list[epoca][layer] > 0, - delta_ij[epoca][layer], np.where(der_list[epoca][layer] < 0, delta_ij[epoca][layer], 0))

                net["W"][layer] = net["W"][layer] + delta_wij[epoca][layer]

            if epoca > 0:
                prod_der = der_list[epoca - 1][layer] * der_list[epoca][layer]

                delta_ij[epoca][layer] = np.where(prod_der > 0, np.minimum(delta_ij[epoca - 1][layer] * eta_plus, delta_max),
                                                  np.where(prod_der < 0, np.maximum(delta_ij[epoca - 1][layer] * eta_minus, delta_min), delta_ij[epoca - 1][layer]))

                delta_wij[epoca][layer] = np.where(prod_der >= 0, - ((np.sign(der_list[epoca][layer])) * delta_ij[epoca][layer]), - delta_wij[epoca - 1][layer])

                net["W"][layer] = net["W"][layer] + delta_wij[epoca][layer]

                der_list[epoca][layer] = np.where(prod_der < 0, 0, der_list[epoca][layer])

        Z_train = forward_propagation(net, X_train)
        err_train = error_function(Z_train, Y_train)

        Z_val = forward_propagation(net, X_val)
        error_val = error_function(Z_val, Y_val)

        print("Epoca: ", epoca, "Train error: ", err_train, "Accuracy Train: ", get_accuracy_network(Z_train, Y_train), "Validation error: ", error_val, "Accuracy Validation: ",
              get_accuracy_network(Z_val, Y_val))

        epoca += 1
