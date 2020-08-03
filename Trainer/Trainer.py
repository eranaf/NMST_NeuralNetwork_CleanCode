
class Trainer:
    def __init__(self, Model, DataReader, Optimizer, typeOfDataToMetricsObject, NumberOfEpoch, SizeOfBatch, derivative_loss):
        """:param epoch: epoch number of epoch in the learning process"""
        # todo: take all the parameters together (NumberOfEpoch and mabye Optimizer...)
        self.info = "number of epoch: " + int.__str__(NumberOfEpoch) + " Size of batch" + int.__str__(SizeOfBatch)

        self.typeOfDataToMetricsObject = typeOfDataToMetricsObject
        self.derivative_loss = derivative_loss
        self.SizeOfBatch = SizeOfBatch
        self.NumberOfEpoch = NumberOfEpoch
        self.Optimizer = Optimizer
        self.DataReader = DataReader
        self.Model = Model
        for metricsObject in self.typeOfDataToMetricsObject.values():
            metricsObject.calc_matrics(self.Model)

    def train(self):
        for epoch in range(self.NumberOfEpoch):
            self.train_step()
            for metricsObject in self.typeOfDataToMetricsObject.values():
                metricsObject.calc_matrics(self.Model)
            print epoch

    def train_step(self):
        for batch_x, batch_target in self.DataReader.get_train_batch_generator(self.SizeOfBatch):
            self.Optimizer.train_step(self.Model, batch_x, batch_target, self.derivative_loss)
        self.DataReader.shuffle_train()

