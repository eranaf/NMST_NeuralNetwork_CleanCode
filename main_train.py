from Data.NMST_Data_Reader import NMST_Data_Reader
from Model.ModelMultiLayeredNetwork import ModelMultiLayeredNetwork
from Predict_functions.accuracy import accuracy, accuracy_percent
from activationFunctions import sigma
import numpy as np



import random as rnd

from losses.losses import loss_nll, loss_mse, derivative_loss_mse
from metrics.metrics import metrics
from optimizers.SGD import SGD
from Trainer.Trainer import Trainer

from Trainer import Trainer
from save_result import save_result


def main(sizes):
    Data_Reader = NMST_Data_Reader()
    Model = ModelMultiLayeredNetwork(sizes=sizes, howToInitialize="normal", activationFunction=sigma.sigma,
                                     activationFunctionDerivative=sigma.derivative_sigma)
    Optimizer = SGD(3.0)
    print Optimizer.__module__
    #metrics
    trainMetrics = metrics(Data_Reader.get_training_data(),(loss_mse, loss_nll,accuracy_percent))
    valMetrics = metrics(Data_Reader.get_validation_data(),(loss_mse, loss_nll,accuracy_percent))
    testMetrics = metrics(Data_Reader.get_test_data(),(loss_mse, loss_nll,accuracy_percent))
    typeOfDataToMetricsObject = {"train": trainMetrics,"validation":valMetrics,"test":testMetrics}

    trainer = Trainer.Trainer(Model, Data_Reader, Optimizer, typeOfDataToMetricsObject, NumberOfEpoch=5, SizeOfBatch=50,
                              derivative_loss=derivative_loss_mse)
    trainer.train()
    save_result(trainer)


    # ### show result or save
    data_train, realOutputs =Data_Reader.get_training_data()
    modelOutputs = Model.feed_forward(data_train)
    print "train"
    print 1.0*accuracy(modelOutputs, realOutputs)/len(np.transpose(modelOutputs))
    data_train, realOutputs = Data_Reader.get_test_data()
    modelOutputs = Model.feed_forward(data_train)
    print "test"
    print 1.0*accuracy(modelOutputs, realOutputs)/len(np.transpose(modelOutputs))




if __name__ == '__main__':
    main([784, 30, 10])



