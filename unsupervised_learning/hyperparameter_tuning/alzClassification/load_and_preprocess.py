#!/usr/bin/env python3
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(data_dir, batch_size=32):
    """
    load the Alzheimer MRI dataset from a specified directory

    Args:
        data_dir (str): directory where the dataset is stored
        batch_size (int): batch size for the data generator
            will default to 32 if not specified when calling the function

    Returns:
        trainingGenerator, validationGenerator: data generators for training and validation sets
    """
    datagen = ImageDataGenerator(validation_split=0.2)  # 20% data for validation

    trainingGenerator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validationGenerator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return trainingGenerator, validationGenerator
