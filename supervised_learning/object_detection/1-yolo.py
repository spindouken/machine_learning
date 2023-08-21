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
                "Anchors must be a numpy.ndarray of shape (outputs, anchor_boxes, 2)."
            )

        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
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

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            tx = output[..., 0:1]
            ty = output[..., 1:2]
            tw = output[..., 2:3]
            th = output[..., 3:4]
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)
            for cy in range(grid_height):
                for cx in range(grid_width):
                    for b in range(anchor_boxes):
                        pw, ph = self.anchors[i][b]
                        bx = (1 / (1 + np.exp(-tx[cy, cx, b]))) + cx
                        by = (1 / (1 + np.exp(-ty[cy, cx, b]))) + cy
                        bw = pw * np.exp(tw[cy, cx, b])
                        bh = ph * np.exp(th[cy, cx, b])
                        bx /= grid_width
                        by /= grid_height
                        bw /= int(self.model.input.shape[1])
                        bh /= int(self.model.input.shape[2])
                        x1 = (bx - (bw / 2)) * image_size[1]
                        y1 = (by - (bh / 2)) * image_size[0]
                        x2 = (bx + (bw / 2)) * image_size[1]
                        y2 = (by + (bh / 2)) * image_size[0]
                        tx[cy, cx, b] = x1
                        ty[cy, cx, b] = y1
                        tw[cy, cx, b] = x2
                        th[cy, cx, b] = y2
            boxes.append(np.concatenate((tx, ty, tw, th), axis=-1))
        return (boxes, box_confidences, box_class_probs)
