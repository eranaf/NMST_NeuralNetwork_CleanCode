from Data.NMST_Data_Reader import NMST_Data_Reader
from Model.ModelMultiLayeredNetwork import ModelMultiLayeredNetwork
import numpy as np


def main():
    D = NMST_Data_Reader()
    data = D.get_train_batch(5)
    M = ModelMultiLayeredNetwork([784, 15, 10], "normal", 20)
    OUTPUT= M.feed_forward(data)
    a=2

if __name__ == '__main__':
    main()