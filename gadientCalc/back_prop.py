import random
import matplotlib.pyplot as plt

import numpy as np


def copy_backpropAll(batch_x, batch_target, sizes, weights, biases, sigma, derivative_sigma, derivative_loss_mse):
    """Update the network's weights and biases by applying
    gradient descent using backpropagation to a single mini batch.
    The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
    is the learning rate."""
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    for x, y in zip(np.transpose(batch_x), np.transpose(batch_target)):
        delta_nabla_w, delta_nabla_b = copy_backprop(x, y, sizes, weights, biases, sigma, derivative_sigma, derivative_loss_mse)
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    return np.array(nabla_w)/len(np.transpose(batch_x)), np.array(nabla_b)/len(np.transpose(batch_x))

def copy_backprop(x, y, sizes, weights, biases, sigma, derivative_sigma, derivative_loss_mse):
    """Return a tuple ``(nabla_b, nabla_w)`` representing the
    gradient for the cost function C_x.  ``nabla_b`` and
    ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    to ``self.biases`` and ``self.weights``."""
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    # feedforward
    activation = x
    activations = [x]  # list to store all the activations, layer by layer
    zs = []  # list to store all the z vectors, layer by layer
    for b, w in zip(biases, weights):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigma(z)
        activations.append(activation)
    # backward pass
    delta = derivative_loss_mse(activations[-1], y) * \
            derivative_sigma(zs[-1])
    nabla_b[-1] = delta
    aaaaaa=np.dot([[x] for x in delta], np.transpose([[x] for x in activations[-2]]))
    nabla_w[-1] = np.dot([[x] for x in delta], np.transpose([[x] for x in activations[-2]]))
    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.
    for l in xrange(2, len(sizes)):
        z = zs[-l]
        sp = derivative_sigma(z)
        delta = np.dot(weights[-l + 1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot([[x] for x in delta], np.transpose([[x] for x in activations[-l - 1]]))
    return nabla_w, nabla_b


def back_prop(batch_x, batch_target, sizes, weights, biases, sigma, derivative_sigma, derivative_loss_mse):

    """" todo: make it better by sending model M with the configurations"""
    z = []
    a = []
    z.append(batch_x)
    a.append(batch_x)
    for weight, bias in zip(weights, biases):
        wx = np.dot(weight, a[-1])
        z.append(wx + bias.reshape(len(bias), 1))
        a.append(sigma(z[-1]))
    delta = [np.zeros([y, len(batch_x)]) for y in sizes[1:]]
    delta[-1] = np.multiply(derivative_loss_mse(a[-1], batch_target), derivative_sigma(z[-1]))
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    nabla_b[-1] = np.average(delta[-1], axis=1)
    nabla_w[-1] = np.divide(np.dot(delta[-1], np.transpose(a[-2])), 1.0 * np.shape(batch_x)[1])
    for l in xrange(2, len(sizes)):
        delta[-l] = np.multiply(np.dot(np.transpose(weights[-l + 1]), delta[-l + 1]), derivative_sigma(z[-l]))
        nabla_b[-l] = np.average(delta[-l], axis=1)
        nabla_w[-l] = np.divide(np.dot(delta[-l], np.transpose(a[-l - 1])), 1.0 * np.shape(batch_x)[1])
    return nabla_w,nabla_b


def debugging_gradient_checking(batch_x, batch_target, sizes, weights, biases, sigma,
                                derivative_sigma, loss_mse, derivative_loss_mse):
    """from NeuralNetworks import NeuralNetworks
...N= NeuralNetworks([784,3,10],"normal",0.1,30)
...from DataReader import DataReader
...D = DataReader()
...batch = D.get_batch('training',100)
...AAA =N.debugging_gradient_checking(batch)
"""
    EPSILON = pow(10, -6)
    ####loss_mse_normal_w_and_b = self.loss_mse(self.feed_forward_matrix(batch_x),batch_target)
    all_place = [(layer, y, x) for layer in range(0, len(sizes) - 1) for x in range(0, sizes[layer])
                 for y in range(0, sizes[layer + 1])]
    random_place = [all_place[x] for x in random.sample(xrange(1, len(all_place)), int(len(all_place) * 0.1))]
    gradient_by_gradient_checking = [calc_gradient_checking(place, EPSILON, batch_x, batch_target,sizes,weights,biases
                                                            ,sigma,loss_mse) for place
                                     in
                                     random_place]
    nabla_w, nabla_b = backpropAll(batch_x, batch_target, sizes, weights, biases, sigma,
                                 derivative_sigma, derivative_loss_mse)
    gradient_back_prob = [nabla_w[place[0]][place[1]][place[2]] for place in random_place]
    relative_error = gradient_by_gradient_checking, gradient_back_prob, np.abs(
        np.divide(np.subtract(gradient_by_gradient_checking, gradient_back_prob),
                  np.add(gradient_by_gradient_checking, gradient_back_prob),
                  out=np.subtract(gradient_by_gradient_checking, gradient_back_prob),
                  where=np.add(gradient_by_gradient_checking,
                               gradient_back_prob)  != 0)), random_place

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, len(relative_error[2])),
            relative_error[2],
            color='#2A6EA6')
    ax.plot(np.arange(0, len(relative_error[2])),
            np.ones(len(relative_error[2])) * np.average(relative_error[2]),
            color='#9B62A6')

    ax.grid(True)
    ax.set_xlabel('n')
    ax.set_title('relative error')
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, len(relative_error[2])),
            np.subtract(gradient_by_gradient_checking, gradient_back_prob),
            color='#2A6EA6')

    ax.grid(True)
    ax.set_xlabel('n')
    ax.set_title('error')
    plt.show()

    return np.subtract(gradient_by_gradient_checking, gradient_back_prob),relative_error, np.average(relative_error[2]), np.max(relative_error[2])


def calc_gradient_checking(place, EPSILON, batch_x, batch_target, sizes, weights, biases, sigma, loss_mse):
    """calc the gradient of the weight in place using the defenition of the derivative"""
    weights[place[0]][place[1]][place[2]] = weights[place[0]][place[1]][place[2]] - EPSILON
    loss_mse_minus = loss_mse(feed_forward(batch_x, weights, biases, sigma), batch_target)
    weights[place[0]][place[1]][place[2]] = weights[place[0]][place[1]][place[2]] + 2 * EPSILON
    loss_mse_plus = loss_mse(feed_forward(batch_x, weights, biases, sigma), batch_target)
    weights[place[0]][place[1]][place[2]] = weights[place[0]][place[1]][place[2]] - EPSILON
    return (loss_mse_plus - loss_mse_minus) / (2 * EPSILON)


def feed_forward(batch_x,weights, biases, sigma):
        """
        feed forward matrix form
        get batch_x (only the input of the network) and return the output of the network
        """
        out_all = batch_x
        for weight, bias in zip(weights, biases):
            wx = np.dot(weight, out_all)
            z = wx + bias.reshape(len(bias), 1)
            out_all = sigma(z)
        # to be equal form to feedforward() -> [np.array(x) for x in np.transpose(out_all)]
        return out_all
































 # # Test 5 - back_prop checking - debuging gradient
    # D = NMST_Data_Reader()
    # batch_x, batch_target = D.get_train_batch(10)
    # network = ModelMultiLayeredNetwork([784, 400, 200, 50, 50, 10], "normal", 20, sigma.sigma)
    #
    # # nabla_wj ~ (C(w + epsilon*e_j) - C(w))/epsilon    *when e_j is the unit vector j
    # # gradient_biases, gradient_weights = network.back_prop(batch)  # should be the real derivative of the w
    # gradient_weights, gradient_biases = backpropAll(batch_x, batch_target, network.sizes, network.w, network.b, sigma.sigma,
    #                                               sigma.derivative_sigma, losses.derivative_loss_mse)
    #
    # epsilon = 10 ** -7
    # gradient_w_avg_relative_error = 0
    # gradient_w_sum_relative_error = 0
    # max_relative_error = 0
    #
    # for j in range(0, (784 * 30 + 30 * 10) // 10):  # 10% of the weights
    #     w_layer = rnd.randint(4, 4)  # random layer
    #     w_neuron = rnd.randint(0, len(network.w[w_layer]) - 1)  # random neuron
    #     w = rnd.randint(0, len(network.w[w_layer][w_neuron]) - 1)  # random weight
    #     # for each weight, will add epsilon for the w only, and calculate (C(w + epsilon*e_j) - C(w))/epsilon
    #     out= network.feed_forward(batch_x)  # normal cost
    #     network.w[w_layer][w_neuron][w] += epsilon
    #     out_epsilon= network.feed_forward(batch_x)  # epsilon cost
    #     network.w[w_layer][w_neuron][w] -= epsilon
    #
    #     # calculating the quadratic cost function for a batch (with or without epsilon)
    #     cost_func_normal = np.sum([losses.loss_mse(a, y) for a,  y in zip(out, batch_target)])
    #     cost_func_epsilon = np.sum([losses.loss_mse(a, y) for a,  y in zip(out_epsilon, batch_target)])
    #
    #     estimated_gradient = (cost_func_epsilon - cost_func_normal) / epsilon/len(np.transpose(batch_x))  # estimated gradient
    #     backprop_gradient = gradient_weights[w_layer][w_neuron][w]  # backpropagation gradient
    #
    #     if backprop_gradient != estimated_gradient:
    #         relative_error = (np.abs(estimated_gradient - backprop_gradient)
    #                           / np.abs(estimated_gradient + backprop_gradient))  # |X-Y|/|X+Y|
    #         gradient_w_sum_relative_error += relative_error
    #         max_relative_error = max(max_relative_error, relative_error)
    #         if j != 0:
    #             gradient_w_avg_relative_error = gradient_w_sum_relative_error / j  # very low number -> estimation and real are close
    #
    # print(gradient_w_avg_relative_error)
    # print(max_relative_error)
    #
    # gradient_weights, gradient_biases = back_prop(batch_x, batch_target, network.sizes, network.w, network.b, sigma.sigma,
    #                                               sigma.derivative_sigma, losses.derivative_loss_mse)
    # backprop_gradient = gradient_weights[w_layer][w_neuron][w]  # backpropagation gradient
    #
    # if backprop_gradient != estimated_gradient:
    #     relative_error = (np.abs(estimated_gradient - backprop_gradient)
    #                       / np.abs(estimated_gradient + backprop_gradient))  # |X-Y|/|X+Y|
    #     gradient_w_sum_relative_error += relative_error
    #     max_relative_error = max(max_relative_error, relative_error)
    #     if j != 0:
    #         gradient_w_avg_relative_error = gradient_w_sum_relative_error / j  # very low number -> estimation and real are close
    #
    # print(gradient_w_avg_relative_error)
    # print(max_relative_error)
