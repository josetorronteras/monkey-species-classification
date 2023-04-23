from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential

class CNNModel(object):

    def __init__(self, config, X):
        """
        Initialize the CNN model
        Arguments:
            config: Configuration file
            X: Data
        """
        self.NB_FILTERS = int(config['CNN_CONFIGURATION']['NB_FILTERS']) 
        self.CONV_SIZE = int(config['CNN_CONFIGURATION']['CONV_SIZE'])
        self.POOL_SIZE = int(config['CNN_CONFIGURATION']['POOL_SIZE'])
        self.NUM_CLASSES = int(config['CNN_CONFIGURATION']['NUM_CLASSES'])
        self.INPUT_SHAPE = X.shape
        
    def build_model(self):
        """
        Build the CNN model
        Return:
            model: CNN model
        """
        model = Sequential()

        model.add(Conv2D(self.NB_FILTERS, (self.CONV_SIZE, self.CONV_SIZE ), input_shape=(150, 150, 3), activation='relu', strides=2))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(self.NB_FILTERS, (self.CONV_SIZE, self.CONV_SIZE ), strides=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(150, activation='relu'))
        model.add(Dense(self.NUM_CLASSES, activation='softmax'))

        return model
        
        