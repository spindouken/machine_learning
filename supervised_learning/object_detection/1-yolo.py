#!/usr/bin/env python3
"""
uses the Yolo v3 algorithm to perform object detection
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """Yolo class to perform object detection using the YOLOv3 algorithm"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        placeholder
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        :param outputs: predictions from Darknet model for a single image
        :param image_size: original size [image_height, image_width]
        :return: tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        imgHeight, imgWidth = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, features = output.shape
            trans_x = output[:, :, :, 0]
            trans_y = output[:, :, :, 1]
            trans_width = output[:, :, :, 2]
            trans_height = output[:, :, :, 3]
            grid_x_pattern = np.arange(grid_width).reshape(1, grid_width, 1)
            grid_x = np.tile(grid_x_pattern, (grid_height, 1, anchor_boxes))
            grid_y_pattern = np.arange(grid_height).reshape(grid_height, 1, 1)
            grid_y = np.tile(grid_y_pattern, (1, grid_width, anchor_boxes))
            box_x = (
                (1 / (1 + np.exp(-trans_x)) + grid_x) * imgWidth / grid_width
            )
            box_y = (
                (1 / (1 + np.exp(-trans_y)) + grid_y) * imgHeight / grid_height
            )
            box_width = (
                np.exp(trans_width)
                * self.anchors[i, :, 0]
                / self.model.input.shape[1]
            ) * imgWidth
            box_height = (
                np.exp(trans_height)
                * self.anchors[i, :, 1]
                / self.model.input.shape[2]
            ) * imgHeight
            x1 = box_x - box_width / 2
            y1 = box_y - box_height / 2
            x2 = box_x + box_width / 2
            y2 = box_y + box_height / 2
            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)
            box_confidences.append(
                1 / (1 + np.exp(-output[..., 4, np.newaxis]))
            )
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))

        return boxes, box_confidences, box_class_probs
