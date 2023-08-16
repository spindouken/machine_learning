#!/usr/bin/env python3
""" Yolo v3 algorithm to perform object detection """

import tensorflow.keras as K
import numpy as np


class Yolo:
    """Yolo class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor"""
        # Load the model
        self.model = K.models.load_model(model_path)
        # Load the classes
        with open(classes_path, "r") as file:
            self.class_names = [line.strip() for line in file.readlines()]
        # Class threshold
        self.class_t = class_t
        # IOU threshold for non-max suppression
        self.nms_t = nms_t
        # Anchor boxes
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet model outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Calculate boxes
            box = np.zeros(output[:, :, :, :4].shape)
            for j in range(grid_height):
                for k in range(grid_width):
                    for l in range(anchor_boxes):
                        tx, ty, tw, th = output[j, k, l, :4]
                        pw, ph = self.anchors[i, l]
                        cx = k
                        cy = j

                        bx = (1 / (1 + np.exp(-tx)) + cx) / grid_width
                        by = (1 / (1 + np.exp(-ty)) + cy) / grid_height
                        bw = pw * np.exp(tw) / self.model.input.shape[1]
                        bh = ph * np.exp(th) / self.model.input.shape[2]

                        x1 = (bx - bw / 2) * image_size[1]
                        y1 = (by - bh / 2) * image_size[0]
                        x2 = (bx + bw / 2) * image_size[1]
                        y2 = (by + bh / 2) * image_size[0]

                        box[j, k, l, 0] = x1
                        box[j, k, l, 1] = y1
                        box[j, k, l, 2] = x2
                        box[j, k, l, 3] = y2

            # Box confidence
            box_confidence = 1 / (1 + np.exp(-output[:, :, :, 4:5]))
            # Box class probabilities
            box_class_prob = 1 / (1 + np.exp(-output[:, :, :, 5:]))

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return (boxes, box_confidences, box_class_probs)
