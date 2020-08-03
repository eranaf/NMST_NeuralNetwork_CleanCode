import numpy as np


class metrics:
    def __init__(self, Data, Metrics_functions):
        """
        :param Metrics_functions - tuple of the fun metrics we wants to calculate
        """
        self.Data = Data
        self.Metrics_functions = Metrics_functions
        self.Metrics_functions_to_Metrics_result = {key: list() for key in Metrics_functions}


    def calc_matrics(self,Model):
        data_train, realOutputs = self.Data
        modelOutputs = Model.feed_forward(data_train)
        for Metrics_function, Metrics_result in self.Metrics_functions_to_Metrics_result.items():
            Metrics_result.append(Metrics_function(modelOutputs, realOutputs))
