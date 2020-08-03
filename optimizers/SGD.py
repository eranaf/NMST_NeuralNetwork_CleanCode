import numpy as np

from gadientCalc.back_prop import back_prop


class SGD:
    def __init__(self, learning_rate):
        """"""
        self.learning_rate = learning_rate
        self.name = "SGD"
        self.info = "learning_rate:" + float.__str__(learning_rate)

    def train_step(self, Model, batch_x, batch_target, derivative_loss):
        nabla_w, nabla_b = back_prop(batch_x, batch_target, Model.sizes, Model.w, Model.b, Model.activationFunction,
                                     Model.activationFunctionDerivative, derivative_loss)
        Model.w = [w - self.learning_rate * n_w for w, n_w in zip(Model.w, nabla_w)]
        Model.b = [b - self.learning_rate * n_b for b, n_b in zip(Model.b, nabla_b)]
