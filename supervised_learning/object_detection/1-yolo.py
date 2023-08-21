#!/usr/bin/env python3
"""
uses the Yolo v3 algorithm to perform object detection
"""
import tensorflow.keras as K
import numpy as np


def sigmoid(x):
    """function to perform sigmoid transformation"""
    return 1 / (1 + np.exp(-x))


def calculate_box(tx, ty, tw, th, pw, ph, grid_width, grid_height, model_shape):
    """placeholder"""
    bx = sigmoid(tx) + np.arange(grid_width).reshape(-1, 1)
    by = sigmoid(ty) + np.arange(grid_height).reshape(1, -1)
    bw = pw * np.exp(tw)
    bh = ph * np.exp(th)
    bx /= grid_width
    by /= grid_height
    bw /= model_shape[1]
    bh /= model_shape[2]
    return bx, by, bw, bh


def process_boxes(output, anchors, image_size, model_shape):
    """placeholder"""
    grid_height, grid_width, num_anchors, _ = output.shape
    boxes = np.zeros((grid_height, grid_width, num_anchors, 4))

    for anchor in range(num_anchors):
        tx, ty, tw, th = output[..., anchor, :4].T
        pw, ph = anchors[anchor]
        bx, by, bw, bh = calculate_box(
            tx, ty, tw, th, pw, ph, grid_width, grid_height, model_shape
        )
        x1 = (bx - (bw / 2)) * image_size[1]
        y1 = (by - (bh / 2)) * image_size[0]
        x2 = (bx + (bw / 2)) * image_size[1]
        y2 = (by + (bh / 2)) * image_size[0]
        boxes[..., anchor, :] = np.stack([x1, y1, x2, y2], axis=-1)

    return boxes


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
        """
        placeholder
        """
        boxes, box_confidences, box_class_probs = [], [], []

        for i, output in enumerate(outputs):
            anchors_for_output = self.anchors[i]
            boxes.append(
                process_boxes(
                    output,
                    anchors_for_output,
                    image_size,
                    self.model.input.shape,
                )
            )
            box_confidences.append(sigmoid(output[..., 4:5]))
            box_class_probs.append(sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs
