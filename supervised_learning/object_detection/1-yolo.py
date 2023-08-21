#!/usr/bin/env python3
"""
uses the Yolo v3 algorithm to perform object detection
"""
import tensorflow as tf
import numpy as np


class Yolo:
    """Yolo class to perform object detection using the YOLOv3 algorithm"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path is the path to where a Darknet Keras model is stored
        classes_path is the path to where the list of class names used
            for the Darknet model, listed in order of index, can be found
        class_t is a float representing the box score
            threshold for the initial filtering step
        nms_t is a float representing the IOU threshold for non-max suppression
        anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
            ...containing all of the anchor boxes:
            outputs is the number of outputs (predictions)
                made by the Darknet model
            anchor_boxes is the number of anchor boxes used for each prediction
            2 => [anchor_box_width, anchor_box_height]
        Public instance attributes:
        model: the Darknet Keras model
        class_names: a list of the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes
        """
        if not tf.keras.models.load_model(model_path):
            raise FileNotFoundError("Model not found at the provided path.")
        if not isinstance(anchors, np.ndarray) or anchors.shape[-1] != 2:
            raise ValueError(
                "Anchors must be a numpy.ndarray \
                of shape (outputs, anchor_boxes, 2)."
            )

        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs from the Darknet model for a single image.
        The nested loops iterate through each grid cell and anchor box to
            compute the processed boundary box coordinates.
        The loops handle the conversion of coordinates from the
            Darknet model's prediction to the
            coordinates relative to the original image.

        outputs: contains predictions from Darknet model for a single image
        i: used to access the corresponding anchor box dimensions
        output: the individual prediction array for processing

        :param outputs: list of numpy.ndarrays containing
            the predictions from the Darknet model for a single image
                Each output will have the shape (grid_height,
                    grid_width, anchor_boxes, 4 + 1 + classes),
                where:
                    - grid_height & grid_width: height
                        and width of the grid used for the output
                    - anchor_boxes: number of anchor boxes used
                    - 4: bounding box coordinates (t_x, t_y, t_w, t_h)
                    - 1: box_confidence
                    - classes: class probabilities for all classes
        :param image_size: numpy.ndarray containing the image’s
            original size [image_height, image_width]
        :return: tuple of (boxes, box_confidences, box_class_probs)
                - boxes: list of numpy.ndarrays of shape (grid_height,
                    grid_width, anchor_boxes, 4) containing the processed
                    boundary boxes for each output, respectively
                4 => (x1, y1, x2, y2) representing
                    the boundary box relative to original image
                - box_confidences: list of numpy.ndarrays of shape
                    (grid_height, grid_width, anchor_boxes, 1)
                    containing the box confidences for each output
                - box_class_probs: list of numpy.ndarrays of shape
                    (grid_height, grid_width, anchor_boxes, classes)
                    containing the box’s class probabilities for each output
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output_index, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            raw_boundary_box_coords = output[..., :4]
            rawBoxConfidence = output[..., 4:5]
            raw_box_class_probabilities = output[..., 5:]

            # Applying sigmoid activation to the box confidence
            box_confidence_after_sigmoid = 1 / (1 + np.exp(-rawBoxConfidence))
            box_confidences.append(box_confidence_after_sigmoid)

            # Applying sigmoid activation to the class probabilities
            box_class_probs_after_sigmoid = 1 / (
                1 + np.exp(-raw_box_class_probabilities)
            )
            box_class_probs.append(box_class_probs_after_sigmoid)

            for cell_y in range(grid_height):
                for cell_x in range(grid_width):
                    for anchor_box_index in range(anchor_boxes):
                        anchor_width, anchor_height = self.anchors[
                            output_index
                        ][anchor_box_index]
                        tx, ty, tw, th = raw_boundary_box_coords[
                            cell_y, cell_x, anchor_box_index
                        ]

                        # Applying sigmoid activation and
                        #   offsetting by grid cell location
                        boundaryBoxCenter_x = (1 / (1 + np.exp(-tx))) + cell_x
                        boundaryBoxCenter_y = (1 / (1 + np.exp(-ty))) + cell_y

                        # Applying exponential and scaling by anchor dimensions
                        boundary_box_width = anchor_width * np.exp(tw)
                        boundary_box_height = anchor_height * np.exp(th)

                        # Normalizing by grid and model input dimensions
                        boundaryBoxCenter_x /= grid_width
                        boundaryBoxCenter_y /= grid_height
                        boundary_box_width /= int(self.model.input.shape[1])
                        boundary_box_height /= int(self.model.input.shape[2])

                        # Converting to original image scale
                        top_left_x = (
                            boundaryBoxCenter_x - (boundary_box_width / 2)
                        ) * image_size[1]
                        top_left_y = (
                            boundaryBoxCenter_y - (boundary_box_height / 2)
                        ) * image_size[0]
                        bottom_right_x = (
                            boundaryBoxCenter_x + (boundary_box_width / 2)
                        ) * image_size[1]
                        bottom_right_y = (
                            boundaryBoxCenter_y + (boundary_box_height / 2)
                        ) * image_size[0]

                        # Storing the processed boundary box coordinates
                        raw_boundary_box_coords[
                            cell_y, cell_x, anchor_box_index
                        ] = [
                            top_left_x,
                            top_left_y,
                            bottom_right_x,
                            bottom_right_y,
                        ]

            boxes.append(raw_boundary_box_coords)

        return (boxes, box_confidences, box_class_probs)
