from numpy import np


def main():
    d = NMST_Data_Reader()
    d.shuffle_train()
    input, output = d.get_train_batch(20)
    from matplotlib import pyplot as plt
    plt.imshow(np.array(np.transpose(input)[3]).reshape(28, 28) * 255, interpolation='nearest')
    plt.show()
    input

if __name__ == '__main__':
    main()