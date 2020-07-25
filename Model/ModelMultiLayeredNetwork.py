import numpy as np



def sigma(z):
    return 1.0 / (1 + np.exp(-z))


class NeuralNetworks:
    def __init__(self, sizes, howToInitialize, epoch):
        """
        :param sizes: [number of inputs, number of neurons in the first hidden layer,., number of neurons in the output layer]
        :param howToInitialize: for now howToInitialize ="normal" - means normal randoms weight and bias
        :param epoch: epoch number of epoch in the learning process
        """
        self.epoch = epoch
        self.sizes = sizes
        self.num_layers = len(sizes)
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
            out_all = sigma(np.add(np.dot(weight, out_all),bias))
        # to be equal form to feedforward() -> [np.array(x) for x in np.transpose(out_all)]
        return out_all


N=NeuralNetworks([784,4,10], "normal", 10)
N
from Data.NMST_Data_Reader import NMST_Data_Reader

D = NMST_Data_Reader()
