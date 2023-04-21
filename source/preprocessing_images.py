from keras.preprocessing.image import ImageDataGenerator

class PreprocessingImages(object):

    def __init__(self, config):
        """
        Initialize the PreprocessingImages class
        Args:
            config: Configuration file
        """
        self.TRAIN_IMAGES_PATH = config['PATH_CONFIGURATION']['TRAIN_IMAGES_PATH']
        self.VALIDATION_IMAGES_PATH = config['PATH_CONFIGURATION']['VALIDATION_IMAGES_PATH']
        self.SIZE = int(config['IMAGE_FEATURES']['SIZE'])

    def preprocessImages(self):
        """
        Preprocess the images
        Args:
            Return:
                train_generator: ImageDataGenerator
                validation_generator: ImageDataGenerator
        """
        # Data augmentation for the training images
        train_datagen = ImageDataGenerator(
            rescale=1./ 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        train_generator = train_datagen.flow_from_directory(
            self.TRAIN_IMAGES_PATH,
            target_size=(self.SIZE, self.SIZE),
            batch_size=64,
            seed=100,
            shuffle=True,
            class_mode='categorical')

        # Data augmentation for the validation images
        validation_datagen = ImageDataGenerator(rescale=1./255)

        validation_generator = validation_datagen.flow_from_directory(
            self.VALIDATION_IMAGES_PATH,
            target_size=(self.SIZE, self.SIZE),
            batch_size=64,
            seed=100,
            shuffle=True,
            class_mode='categorical')
        
        return train_generator, validation_generator