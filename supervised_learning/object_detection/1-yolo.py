#!/usr/bin/env python3
"""This module utilizes the YOLO v3 algorithm for
object detection using the Darknet model."""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """Yolo class to perform object detection using the YOLOv3 algorithm."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initializes the Yolo object with the model,
        classes, thresholds, and anchors."""
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Processes the outputs from the Darknet
        model to obtain bounding boxes,
        box confidences, and class probabilities."""
        boxes = []
        box_confidences = []
        box_class_probs = []
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]
        image_height, image_width = image_size[0], image_size[1]
        corner_x, corner_y = (
            [],
            [],
        )
        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes = output.shape[:3]
            box = output[..., :4]
            grid_x = np.arange(grid_width).reshape(1, grid_width)
            grid_x = np.repeat(grid_x, grid_height, axis=0)
            grid_y = np.arange(grid_width).reshape(1, grid_width)
            grid_y = np.repeat(grid_y, grid_height, axis=0).T
            corner_x.append(
                np.repeat(grid_x[..., np.newaxis], anchor_boxes, axis=2)
            )
            corner_y.append(
                np.repeat(grid_y[..., np.newaxis], anchor_boxes, axis=2)
            )
            boundary_x = (sigmoid(box[..., 0]) + corner_x[i]) / output.shape[1]
            boundary_y = (sigmoid(box[..., 1]) + corner_y[i]) / output.shape[0]
            boundary_w = (
                np.exp(box[..., 2]) * self.anchors[i, :, 0]
            ) / input_width
            boundary_h = (
                np.exp(box[..., 3]) * self.anchors[i, :, 1]
            ) / input_height
            box[..., 0] = (boundary_x - (boundary_w * 0.5)) * image_width
            box[..., 1] = (boundary_y - (boundary_h * 0.5)) * image_height
            box[..., 2] = (boundary_x + (boundary_w * 0.5)) * image_width
            box[..., 3] = (boundary_y + (boundary_h * 0.5)) * image_height
            boxes.append(box)
            box_confidences.append(sigmoid(output[..., 4:5]))
            box_class_probs.append(sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function to transform
        values to the range (0, 1)."""
        return 1 / (1 + np.exp(-x))
