#!/usr/bin/env python3
"""
uses the Yolo v3 algorithm to perform object detection
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
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
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Iterates through each output in 'outputs' using enumerate
            to get both the index 'i' and the value 'output'

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
            cell_width = image_size[1] / grid_width
            cell_height = image_size[0] / grid_height
            box = output[..., :4]
            box[..., :2] = 1 / (1 + np.exp(-box[..., :2]))
            box[..., 2:] = np.exp(box[..., 2:])
            box[..., 0] += np.arange(grid_width).reshape(1, grid_width, 1)
            box[..., 1] += np.arange(grid_height).reshape(grid_height, 1, 1)
            box[..., :2] /= (grid_width, grid_height)
            box[..., 2:] *= self.anchors[i]
            box[..., 0] -= box[..., 2] / 2
            box[..., 1] -= box[..., 3] / 2
            box[..., 2] += box[..., 0]
            box[..., 3] += box[..., 1]
            box[..., [0, 2]] *= cell_width
            box[..., [1, 3]] *= cell_height
            boxes.append(box)

            aux = output[:, :, :, 4]
            conf = 1 / (1 + np.exp(-aux))
            conf = conf.reshape(grid_height, grid_width, anchor_boxes, 1)
            box_confidences.append(conf)

            aux = output[:, :, :, 5:]
            probs = 1 / (1 + np.exp(-aux))
            box_class_probs.append(probs)

        return boxes, box_confidences, box_class_probs
