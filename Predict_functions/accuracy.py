import numpy as np


def accuracy(modelOutputs, realOutputs):
    modelOutputsNumber = np.argmax(modelOutputs, axis=0)
    realOutputsNumber = np.argmax(realOutputs, axis=0)
    return sum(modelOutputsNumber == realOutputsNumber)


def accuracy_percent(modelOutputs, realOutputs):
    return 100.0*accuracy(modelOutputs, realOutputs)/len(np.transpose(modelOutputs))