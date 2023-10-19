from  source.preprocessing_images import PreprocessingImages
import configparser

# Cargar configuración para pruebas
config = configparser.ConfigParser()
config.read('config.ini')

def test_preprocessing_images():
    preprocessing = PreprocessingImages(config['CONFIGURATION'])
    train_generator, validation_generator = preprocessing.preprocessImages()
    
    assert train_generator is not None
    assert validation_generator is not None
    assert train_generator.batch_size == 64
    assert validation_generator.batch_size == 64
    assert train_generator.class_mode == 'categorical'
    assert validation_generator.class_mode == 'categorical'