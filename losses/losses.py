import numpy as np


def loss_mse(modelOutputs, realOutputs):
    diff = np.subtract(modelOutputs, realOutputs)
    norm = np.linalg.norm(diff, 2, 0)
    AA=norm.shape
    if norm.shape != np.array(1).shape:
        normOfAll = np.linalg.norm(norm, 2)
    else:
        normOfAll = norm
    result = normOfAll * normOfAll / (norm.size * 2)
    """ Equals to:
    norm2 = 0.5*np.multiply(norm, norm)
    result = np.average(norm2)"""
    return result


def derivative_loss_mse(modelOutputs, realOutputs):
    # vector of dC/da(i)
    return np.subtract(modelOutputs, realOutputs)
    #return modelOutputs - realOutputs

def loss_nll(modelOutputs, realOutputs):
    # todo Show that the negative log loss over a batch (of i.i.d examples) of size n goes to the cross entropy
    #  between PY |X and QY |X (the true conditional distribution and the neural network model respectively)
    #  as n -> infinity
    #  why it is
    log_output = np.log(modelOutputs)
    nll_per_input = -np.nansum(np.multiply(realOutputs, log_output), 0)
    result = np.average(nll_per_input)
    return result
