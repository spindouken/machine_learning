#!/usr/bin/env python3
"""
Use transfer learning to build on a pre-trained CNN
    model for Alzheimer's classification
"""
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    Input,
    concatenate,
)
from tensorflow.keras.applications import DenseNet169


def create_model(input_shape, num_classes):
    """
    Create the custom CNN model for Alzheimer's classification

    Parameters:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of classes in the dataset.
    """
    # pre-trained DenseNet169 as the base model
    base_model = DenseNet169(
        include_top=False, weights="imagenet", input_shape=input_shape
    )

    # freeze all layers except the last two dense blocks
    for layer in base_model.layers[:-17]:
        layer.trainable = False

    # get the output tensor of the base model
    base_model_output = base_model.output

    x = GlobalAveragePooling2D()(base_model_output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(num_classes, activation="softmax")(x)

    # compile Model
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
