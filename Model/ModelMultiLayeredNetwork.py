import numpy as np


class ModelMultiLayeredNetwork:
    def __init__(self, sizes, howToInitialize, activationFunction,activationFunctionDerivative):
        """
        :param sizes: [number of inputs, number of neurons in the first hidden layer,., number of neurons in the output layer]
        :param howToInitialize: for now howToInitialize ="normal" - means normal randoms weight and bias
        """
        self.name = "NeuralNetwork"
        self.info = "sizes: " + list.__str__(sizes)

        self.sizes = sizes
        self.num_layers = len(sizes)
        self.activationFunction = activationFunction
        self.activationFunctionDerivative = activationFunctionDerivative
        if howToInitialize == "normal":
            self.w = [np.random.randn(y, x) for x, y in zip(sizes[0:-1], sizes[1:])]
            self.b = [np.random.randn(y) for y in sizes[1:]]

    def feed_forward(self, batch_x):
        """
        feed forward matrix form
        get batch_x (only the input of the network) and return the output of the network
        """
        out_all = batch_x
        for weight, bias in zip(self.w, self.b):
            wx = np.dot(weight, out_all)
            z = wx + bias.reshape(len(bias), 1)
            out_all = self.activationFunction(z)
        # to be equal form to feedforward() -> [np.array(x) for x in np.transpose(out_all)]
        return out_all

