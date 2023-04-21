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
        self.NB2_FILTERS = int(config['CNN_CONFIGURATION']['NB2_FILTERS'])
        self.CONV1_SIZE = int(config['CNN_CONFIGURATION']['CONV1_SIZE'])
        self.CONV2_SIZE = int(config['CNN_CONFIGURATION']['CONV2_SIZE'])
        self.NUM_CLASSES = int(config['CNN_CONFIGURATION']['NUM_CLASSES'])
        self.INPUT_SHAPE = X.shape
        
    def build_model(self):
        """
        Build the CNN model
        Return:
            model: CNN model
        """
        model = Sequential()
        model.add(Conv2D(32, (self.CONV1_SIZE , self.CONV1_SIZE ), activation='relu', input_shape=self.INPUT_SHAPE))
        model.add(Conv2D(32, (self.CONV1_SIZE, self.CONV1_SIZE), activation='relu'))
        model.add(MaxPooling2D(pool_size=(self.CONV2_SIZE , self.CONV2_SIZE )))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (self.CONV1_SIZE, self.CONV1_SIZE), activation='relu'))
        model.add(Conv2D(64, (self.CONV1_SIZE, self.CONV1_SIZE), activation='relu'))
        model.add(MaxPooling2D(pool_size=(self.CONV2_SIZE , self.CONV2_SIZE )))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.NUM_CLASSES, activation='softmax'))
        
        return model
        
        