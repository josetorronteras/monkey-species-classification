import pytest
import numpy as np
from source.cnn_model import CNNModel

@pytest.fixture
def mock_data():
    num_classes = 2
    num_images_per_class = 50
    image_size = (24, 24)

    class1_images = np.random.randint(0, 256, size=(num_images_per_class, *image_size), dtype=np.uint8)
    class2_images = np.random.randint(0, 256, size=(num_images_per_class, *image_size), dtype=np.uint8)

    class1_images = np.expand_dims(class1_images, axis=-1)
    class2_images = np.expand_dims(class2_images, axis=-1)

    X = np.concatenate([class1_images, class2_images], axis=0)

    return X

@pytest.mark.parametrize("config", [
    (
        {
            'CNN_CONFIGURATION': {
                'NB_FILTERS': 32,
                'CONV_SIZE': 3,
                'POOL_SIZE': 2,
                'NUM_CLASSES': 10
            }
        }
    ),
])
def test_model_creation(config, mock_data):
    cnn_model = CNNModel(config, mock_data)

    try:
        model = cnn_model.build_model()
        assert model is not None
    except Exception as e:
        pytest.fail(f'Error build model: {e}')

@pytest.mark.parametrize("config", [
    (
        {
            'CNN_CONFIGURATION': {
                'NB_FILTERS': 32,
                'CONV_SIZE': 3,
                'POOL_SIZE': 2,
                'NUM_CLASSES': 10
            }
        }
    ),
])
def test_model_compilation(config, mock_data):
    cnn_model = CNNModel(config, mock_data)
    model = cnn_model.build_model()

    try:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        assert len(model.layers) > 0
    except Exception as e:
        pytest.fail(f'Error compiling model: {e}')