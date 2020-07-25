import gzip
import numpy as np

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10,))
    e[j] = 1.0
    return e

class NMST_Data_Reader:
    def __init__(self):
        """ Intializing the data reader object.
               55k train with labels,
               5k valid with labels
               and 10k test with labels
               train positon"""
        print ('Initializing data reader object...')
        data_Train_Images, data_Train_Labels, data_Test_Image, data_Test_Labels = self.readDataFromFile()
        test_10k_x, test_10k_y, training_55k_x, training_55k_y, validation_5k_x, validation_5k_y = self.dataTransform(
            data_Test_Image, data_Test_Labels, data_Train_Images, data_Train_Labels)
        self.train = zip(training_55k_x, training_55k_y)
        self.valid = zip(validation_5k_x, validation_5k_y)
        self.test = zip(test_10k_x, test_10k_y)

        self.train_position = 0
        print ('Initialized!')

    def get_train_batch(self, size):
        """return a batch in size "size" from the type train, every time it increase the place we take from.
        when get to the edge and ask for another batch' it send you a message to suffel
        when it before the end but have less then 'size' element it return need to shuffle"""
        if self.train_position + size >= len(self.train):
            return 'need to shuffle'
        else:
            batch = self.train[self.train_position:self.train_position + size]
            self.train_position = self.train_position + size
        return self.unzip_batch(batch)

    def shuffle_train(self):
        np.random.shuffle(self.train)
        self.train_position = 0

    def get_training_data(self):
        """
        :return: all the training data and it's target
        """
        return self.unzip_batch(self.train)

    def get_validation_data(self):
        """return all the validation data and it's target"""
        return self.unzip_batch(self.valid)

    def get_test_data(self):
        """
        :return:  all the test data and it's target
        """
        return self.unzip_batch(self.test)

    @staticmethod
    def unzip_batch(batch):
        """unzip the batch into a input matrix and output matrix"""
        unzip = [[i for i, j in batch],
                 [j for i, j in batch]]
        return np.transpose(unzip[0]), np.transpose(unzip[1])

    def dataTransform(self, data_Test_Image, data_Test_Labels, data_Train_Images, data_Train_Labels):
        """transform the data to vector of enter from 0 to 1 in size of 28*28 and label vector in size 10"""
        training_55k_x = data_Train_Images[0:55000] / 255.  # vector normalized
        validation_5k_x = data_Train_Images[55000: 60000] / 255.  # vector normalized
        training_55k_y = self.result_to_vector(data_Train_Labels[0:55000])
        validation_5k_y = self.result_to_vector(data_Train_Labels[55000: 60000])
        test_10k_x = data_Test_Image[0: 10000] / 255.  # vector normalized
        test_10k_y = self.result_to_vector(data_Test_Labels[0: 10000])
        return test_10k_x, test_10k_y, training_55k_x, training_55k_y, validation_5k_x, validation_5k_y

    @staticmethod
    def readDataFromFile():
        """read the data from the file"""
        image_size = 28  # each image is 28x28

        num_images = 60000  # there are 60k images
        with gzip.open('train-images-idx3-ubyte.gz', 'r') as f: # 60k train & valid
            f.read(16)  # reading by 16-byte double
            buffer_Train_Images = f.read(image_size * image_size * num_images)
            f.close()
            data_Train_Images = np.frombuffer(buffer_Train_Images, dtype=np.uint8).astype(np.int32)  # translating into 0 to 255
            data_Train_Images = data_Train_Images.reshape(num_images, image_size * image_size)  # data = 60k x 28 x 28 with 1 value in it

        with gzip.open('train-labels-idx1-ubyte.gz', 'r') as f:  # 60k train & valid - labels
            f.read(8)  # reading by 16-byte double
            buffer_Train_Labels = f.read(num_images)
            data_Train_Labels = np.frombuffer(buffer_Train_Labels, dtype=np.uint8).astype(np.int32)  # translating into 0 to 255

        num_images = 10000  # there are 10k images
        with gzip.open('t10k-images-idx3-ubyte.gz', 'r') as f: # 10k tests
            f.read(16)  # reading by 16-byte double
            buffer_Test_Image = f.read(image_size * image_size * num_images)
            data_Test_Image = np.frombuffer(buffer_Test_Image, dtype=np.uint8).astype(np.uint8)  # translating into 0 to 255
            data_Test_Image = data_Test_Image.reshape(num_images, image_size * image_size)  # data = 60k x 28 x 28 with

        with gzip.open('t10k-labels-idx1-ubyte.gz', 'r') as f:  # 10k tests - lbles
            f.read(8)  # reading by 16-byte double
            buffer_Test_Label = f.read(num_images)
            data_Test_Labels = np.frombuffer(buffer_Test_Label, dtype=np.uint8).astype(np.int32)  # translating into 0 to 255

        return data_Train_Images,data_Train_Labels,data_Test_Image,data_Test_Labels

    @staticmethod
    def result_to_vector(results):
        """take number 0-9 and transform it to a vector with one in the place of results [0..1..0]"""
        return [vectorized_result(x) for x in results]

d = NMST_Data_Reader()
d.shuffle_train()
input,output= d.get_train_batch(20)
from matplotlib import pyplot as plt
plt.imshow(np.array(np.transpose(input)[3]).reshape(28,28)*255, interpolation='nearest')
plt.show()
input