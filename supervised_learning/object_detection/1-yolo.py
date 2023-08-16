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
            # Extracting grid_height and grid_width from the output shape
            #   The third and fourth dimensions are 'anchor_boxes' and
            #       (combined) 'features' (4 + 1 + classes)
            #       and are not needed in the subsequent calculations,
            #       so they are ignored
            grid_height, grid_width, anchor_boxes, features = output.shape

            # cell_width and cell_height are calculated
            #   by dividing image’s original height and width
            #   by corresponding height and width of the grid
            # ...this will be used to resize bounding boxes later
            cell_width = image_size[1] / grid_width
            cell_height = image_size[0] / grid_height

            # DECODING BOUNDING BOXES
            # extract first four elements along last dimension of the tensor
            #   In YOLO the output tensor's last dimension includes information
            #       about bounding box coordinates tx, ty, tw, th
            #       confidence score, and class probabilities
            #   By slicing :4 we are extracting the bounding box coordinates
            box = output[..., :4]

            # Apply sigmoid to x,y coordinates (1st two elements in last dim)
            #   sigmoid will constrain the values to range (0, 1)
            box[..., :2] = 1 / (1 + np.exp(-box[..., :2]))  # x, y

            # Apply exp to width and height (last two elements in last dim)
            #   ...During prediction process (training of Darknet),
            #   the logarithm of the width and height was taken
            #   ...applying the exponential will reverse that transformation
            box[..., 2:] = np.exp(box[..., 2:])  # w, h

            # Add grid indices to x, y coordinates
            box[..., 0] += np.arange(grid_width).reshape(1, grid_width, 1)  # x
            box[..., 1] += np.arange(grid_height).reshape(grid_height, 1, 1)

            # Divide x, y coordinates by grid height and grid width
            #   ...this will normalize the coordinates to range (0, 1)
            box[..., :2] /= (grid_width, grid_height)  # x, y

            # Multiply w, h coordinates by anchors
            #   this will scale the coordinates to the dims of the anchors
            #   ...the anchors are given in width, height format
            #   ...the width and height of anchors are scaled to the grid dims
            #   ...the result will be bounding boxes with width and height
            #   ...that are relative to the dimensions of the grid
            box[..., 2:] *= self.anchors[i]  # w, h

            # Calc top-left and bottom-right coordinates of the bounding box
            box[..., 0] -= box[..., 2] / 2  # x1 = x - w/2
            box[..., 1] -= box[..., 3] / 2  # y1 = y - h/2
            box[..., 2] += box[..., 0]  # x2 = x1 + w
            box[..., 3] += box[..., 1]  # y2 = y1 + h

            # resize bounding boxes to match original image size
            box[..., [0, 2]] *= cell_width  # Scaling x coordinates
            box[..., [1, 3]] *= cell_height  # Scaling y coordinates

            # Append boxes, box_confidences, and box_class_probs
            #   to their respective lists
            boxes.append(box)
            # apply sigmoid to box confidence scores&append to box_confidences
            box_confidences.append(
                1 / (1 + np.exp(-output[..., 4, np.newaxis]))
            )
            # apply sigmoid to box class scores and append to box_class_probs
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))

        # return a tuple of (boxes, box_confidences, box_class_probs)
        return boxes, box_confidences, box_class_probs
