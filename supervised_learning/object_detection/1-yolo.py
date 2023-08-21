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
        with open(classes_path, "r") as file:
            self.class_names = [line.strip() for line in file.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            box, box_confidence, box_class_prob = self.process_single_output(
                output, i, image_size
            )
            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def process_single_output(self, output, i, image_size):
        grid_height, grid_width, anchor_boxes, _ = output.shape
        box = np.zeros((grid_height, grid_width, anchor_boxes, 4))

        bx, by, bw, bh = self.calculate_box_coordinates(
            output, grid_width, grid_height, i
        )

        # Updated the call to include grid_width and grid_height
        x1, y1, x2, y2 = self.calculate_image_coordinates(
            bx, by, bw, bh, image_size, grid_width, grid_height
        )

        box[..., 0] = x1
        box[..., 1] = y1
        box[..., 2] = x2
        box[..., 3] = y2

        box_confidence = sigmoid(output[..., 4])
        box_confidence = box_confidence.reshape(
            grid_height, grid_width, anchor_boxes, 1
        )

        box_class_prob = sigmoid(output[..., 5:])

        return box, box_confidence, box_class_prob

    def calculate_box_coordinates(self, output, grid_width, grid_height, i):
        t_x = output[..., 0]
        t_y = output[..., 1]
        t_w = output[..., 2]
        t_h = output[..., 3]
        pw = self.anchors[i, :, 0]
        ph = self.anchors[i, :, 1]
        bx = sigmoid(t_x) + np.arange(grid_width).reshape(1, grid_width, 1)
        by = sigmoid(t_y) + np.arange(grid_height).reshape(grid_height, 1, 1)
        bw = pw * np.exp(t_w)
        bh = ph * np.exp(t_h)
        return bx, by, bw, bh

    def calculate_image_coordinates(
        self, bx, by, bw, bh, image_size, grid_width, grid_height
    ):
        bx /= grid_width
        by /= grid_height
        bw /= self.model.input.shape[1].value
        bh /= self.model.input.shape[2].value
        x1 = (bx - bw / 2) * image_size[1]
        y1 = (by - bh / 2) * image_size[0]
        x2 = (bx + bw / 2) * image_size[1]
        y2 = (by + bh / 2) * image_size[0]
        return x1, y1, x2, y2


def sigmoid(x):
    """function to perform sigmoid transformation"""
    return 1 / (1 + np.exp(-x))
