#!/usr/bin/env python3
"""
This module utilizes the YOLO v3 algorithm for object detection
using the Darknet model.
It consists of a Yolo class that contains
methods for initializing and processing the model's outputs.
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Yolo class to perform object detection using the YOLOv3 algorithm.
    It encapsulates the Darknet Keras model and provides methods for
    initialization and output processing.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo object with the model,
            classes, thresholds, and anchors.

        Public instance attributes:
            model: the Darknet Keras model
            class_names: a list of the class names for the model
            class_t: the box score threshold for the initial filtering step
            nms_t: the IOU threshold for non-max suppression
            anchors: the anchor boxes
                numpy.ndarray of shape (outputs, anchor_boxes, 2)
                ...containing all of the anchor boxes:
                    outputs is the number of outputs (predictions)
                        made by the Darknet model
                    anchor_boxes is # of anchor boxes used for each prediction
                        2 => [anchor_box_width, anchor_box_height]

        Args:
            model_path (str): Path to the Darknet Keras model.
            classes_path (str): Path to the class names.
            class_t (float): Box score threshold.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Anchor boxes.
        """
        # Load the Keras model from the given path
        self.model = K.models.load_model(model_path)
        # Read and store class names from the given file path
        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        # Store the class score threshold, IOU threshold, and anchor boxes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs from the Darknet model to obtain bounding boxes,
            box confidences, and class probabilities.
        Put simply, it takes the raw predictions and
            transforms them into a more interpretable format.

        1. Initialization: Empty lists are initialized for boxes, confidences,
            and class probabilities
            ...Input width, height, and image size are extracted

        2. Grid Cell Indexing: Iterates through outputs,
            extracting grid height, width, and anchor boxes
            ...Creates grid cell indices for x, y coordinates

        3. Bounding Box Decoding: Decodes bounding box coordinates
            using sigmoid and exponential transformation
            ...Calculates boundary x, y, width, height

        4. Bounding Box Resizing: Resizes bounding boxes to match
            original image size
            ...Scales x, y coordinates by image's width, height

        5. Confidence Score and Class Probabilities:
            Applies sigmoid to confidence scores and class probabilities,
                constraining values to range (0, 1)

        6. Appending Results: Appends processed bounding boxes,
            confidence scores, and class probabilities to respective lists

        7. Return: Returns a tuple with lists of boxes,
            box confidences, and box class probabilities

        Args:
            outputs (list): Predictions from Darknet model for an image.
                            Each output shape: (grid_height, grid_width,
                                anchor_boxes, 4 + 1 + classes)
            image_size (numpy.ndarray): Original size of image [image_height,
                                        image_width]

        Returns:
            tuple: Processed boxes, box confidences, box class probabilities
                Boxes relative to original image size,
                    confidences and class probabilities range (0, 1)
        """
        # Initialize lists for processed outputs
        boxes = []  # List to store processed bounding boxes
        box_confidences = []  # List to store box confidence scores
        box_class_probs = []  # List to store box class probabilities

        # Extract input width and height from the model's input shape
        # model.input.shape = (batch_size, input_width, input_height, channels)
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]
        # Extract image height and width from the given image size
        image_height, image_width = image_size[0], image_size[1]
        corner_x, corner_y = (
            [],
            [],
        )  # Lists to store grid cell indices for x and y

        # Iterate through each output (prediction) from the Darknet model
        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes = output.shape[
                :3
            ]  # Extract grid dimensions and number of anchor boxes from output
            box = output[..., :4]  # Extract bounding box coordinates

            # Create grid cell indices for x and y using numpy broadcasting
            #   repeat functions
            grid_x = np.arange(grid_width).reshape(1, grid_width)
            grid_x = np.repeat(grid_x, grid_height, axis=0)
            grid_y = np.arange(grid_width).reshape(1, grid_width)
            # transpose grid_y to make it a column vector that
            #   matches grid_x's shape
            grid_y = np.repeat(grid_y, grid_height, axis=0).T

            # Append grid cell indices to corner_x and corner_y
            #   using numpy broadcasting
            # corner_x and corner_y will be used to calculate
            #   the bounding box coordinates
            corner_x.append(
                np.repeat(grid_x[..., np.newaxis], anchor_boxes, axis=2)
            )
            corner_y.append(
                np.repeat(grid_y[..., np.newaxis], anchor_boxes, axis=2)
            )

            # DECODING BOUNDING BOXES
            # Decode x and y coordinates using sigmoid & add grid cell indices
            boundary_x = (sigmoid(box[..., 0]) + corner_x[i]) / output.shape[1]
            boundary_y = (sigmoid(box[..., 1]) + corner_y[i]) / output.shape[0]
            # Decode width and height using exponential
            #   transformation and scale by anchors
            boundary_w = (
                np.exp(box[..., 2]) * self.anchors[i, :, 0]
            ) / input_width
            boundary_h = (
                np.exp(box[..., 3]) * self.anchors[i, :, 1]
            ) / input_height

            # Calculate bounding box coordinates of the top-left (x1, y1) and
            #   bottom-right (x2, y2) and resize to original image size
            box[..., 0] = (boundary_x - (boundary_w * 0.5)) * image_width  # x1
            box[..., 1] = (boundary_y - (boundary_h * 0.5)) * image_height  # y
            box[..., 2] = (boundary_x + (boundary_w * 0.5)) * image_width  # x2
            box[..., 3] = (boundary_y + (boundary_h * 0.5)) * image_height  # y

            # Append processed bounding boxes, confidence scores,
            #   and class probabilities
            boxes.append(box)
            box_confidences.append(
                sigmoid(output[..., 4:5])
            )  # Apply sigmoid to confidence scores
            box_class_probs.append(
                sigmoid(output[..., 5:])
            )  # Apply sigmoid to class probabilities

        return boxes, box_confidences, box_class_probs


def sigmoid(x):
    """
    Sigmoid activation function to transform values to the range (0, 1).

    Args:
        x (float): Input value.

    Returns:
        float: Transformed value in the range (0, 1).
    """
    return 1 / (1 + np.exp(-x))
