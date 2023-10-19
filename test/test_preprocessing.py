import pytest
from source.preprocessing_images import PreprocessingImages
import os

@pytest.mark.parametrize("config", [
    (
        {
            'PATH_CONFIGURATION': {
                'TRAIN_IMAGES_PATH': 'test/images/training',
                'VALIDATION_IMAGES_PATH': 'test/images/validation'
            },
            'IMAGE_FEATURES': {
                'SIZE': '150'
            }
        }
    ),
])
def test_preprocessing_images(config):
    preprocessing = PreprocessingImages(config)
    train_generator, validation_generator = preprocessing.preprocessImages()

    assert train_generator is not None
    assert validation_generator is not None