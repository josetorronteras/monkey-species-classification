import pytest
from ..source.cnn_model import CNNModel

def test_model_creation():
    config = {
        'CNN_CONFIGURATION': {
            'NB_FILTERS': 32,
            'CONV_SIZE': 3,
            'POOL_SIZE': 2,
            'NUM_CLASSES': 10
        }
    }
    
    mock_image_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mock_batch_data = [mock_image_data]
    mock_generator_data = [[mock_batch_data]]

    cnn_model = CNNModel(config, mock_generator_data)

    try:
        model = cnn_model.build_model()
        assert model is not None
    except Exception as e:
        pytest.fail(f'Error build model: {e}')


def test_model_compilation():
    model = CNNModel(config, train_generator[0][0][0]).build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    assert len(model.layers) > 0