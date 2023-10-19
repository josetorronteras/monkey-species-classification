from source.cnn_model import CNNModel, train_generator
import configparser

config = configparser.ConfigParser()
config.read('testing-config.ini')

def test_model_creation():
    model = CNNModel(config, train_generator[0][0][0]).build_model()
    assert model is not None

def test_model_compilation():
    model = CNNModel(config, train_generator[0][0][0]).build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    assert len(model.layers) > 0