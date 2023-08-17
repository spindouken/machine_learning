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
            to get both the index 'i' and the value 'output'.
        Then, it calculates the transformed bounding box
            coordinates (x1, y1, x2, y2) based on the predicted transformations
        The grid cell indices are created and used to
            calculate the final processed bounding box parameters.
        Sigmoid and exponential functions are applied where necessary to
            convert the raw outputs into interpretable values.

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

        imgHeight, imgWidth = image_size

        for i, output in enumerate(outputs):
            # Extracting the dimensions of the grid
            # -----------------------------------------------------------------
            #  These are used to normalize the bounding box coordinates
            # 'features' not currently being utilized
            grid_height, grid_width, anchor_boxes, features = output.shape

            # Extracting predicted transformations for bounding boxes from
            #   the 'output' tensor
            # -----------------------------------------------------------------
            # These transformations are relative changes that need to be
            #   applied to the anchor boxes to predict the actual
            #   ...bounding boxes for detected objects.
            # - trans_x and trans_y: relative shift from anchor box center;
            #   often passed through a sigmoid function to constrain values.
            # - trans_width and trans_height: relative scaling of anchor box's
            #   width and height; transformed using exponential function to
            #   map from logarithmic to linear scale.
            # Together, these transformations provide the necessary adjustments
            # to anchor boxes to accurately encapsulate detected objects
            trans_x = output[:, :, :, 0]
            trans_y = output[:, :, :, 1]
            trans_width = output[:, :, :, 2]
            trans_height = output[:, :, :, 3]

            # Creating grid indices to map the
            #   spatial location of each cell in the grid
            # -----------------------------------------------------------------
            # These indices are essential for aligning
            #   bounding boxes with their spatial context

            # Creating x grid pattern and tiling to match grid dimensions
            grid_x_pattern = np.arange(grid_width).reshape(1, grid_width, 1)
            grid_x = np.tile(grid_x_pattern, (grid_height, 1, anchor_boxes))

            # Creating y grid pattern and tiling to match grid dimensions
            grid_y_pattern = np.arange(grid_height).reshape(grid_height, 1, 1)
            grid_y = np.tile(grid_y_pattern, (1, grid_width, anchor_boxes))

            # Calculating processed x and y coordinates by applying sigmoid
            # -----------------------------------------------------------------
            # Adding grid indices ensures correct
            #   positioning of bounding boxes in the image
            # Scaling factors are applied to match original image's dimensions
            box_x = (
                (1 / (1 + np.exp(-trans_x)) + grid_x) * imgWidth / grid_width
            )
            box_y = (
                (1 / (1 + np.exp(-trans_y)) + grid_y) * imgHeight / grid_height
            )

            # Calculating processed x and y coordinates
            #   by applying sigmoid (to constrain values between 0 and 1)
            # -----------------------------------------------------------------
            # Adding grid indices helps to position
            #   the bounding boxes correctly within the image
            box_x = (
                (1 / (1 + np.exp(-trans_x)) + grid_x) * imgWidth / grid_width
            )
            box_y = (
                (1 / (1 + np.exp(-trans_y)) + grid_y) * imgHeight / grid_height
            )

            # Calculating processed width & height using exponential function;
            #   reverse logarithm taken during prediction (training of Darknet)
            # -----------------------------------------------------------------
            # The anchors are scaled to reflect the actual proportions
            #   of the bounding boxes in the original image
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

            # Compute the top-left (x1, y1)
            #   and bottom-right (x2, y2) coordinates
            #   ...of the bounding boxes to define their boundaries
            # -----------------------------------------------------------------
            # These coordinates are relative to the original image
            #   and can be used for drawing the bounding boxes

            # Calculate x1, the x-coordinate of the left edge of bounding box
            #   by subtracting half the width from x coordinate of the center
            x1 = box_x - box_width / 2

            # Calculate y1, the y-coordinate of the top edge of bounding box
            #   by subtracting half the height from y coordinate of the center
            y1 = box_y - box_height / 2

            # Calculate x2, the x-coordinate of the right edge of bounding box
            #   by adding half the width to the x coordinate of the center
            x2 = box_x + box_width / 2

            # Calculate y2, the y-coordinate of the bottom edge of bounding box
            #   by adding half the height to the y coordinate of the center
            y2 = box_y + box_height / 2

            # We will use these coordinates later with OpenCV
            #   to draw the bounding boxes on images
            # Together, (x1, y1) gives the coordinates of the top-left corner,
            #   and (x2, y2) gives the coordinates of the bottom-right corner
            #   ...of the bounding box

            # APPENDING 'boxes', 'box_confidences', 'box_class_probs'
            # -----------------------------------------------------------------
            # Appending the bounding box coordinates to the 'boxes' list
            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            # Applying sigmoid to box confidence scores
            #   to constrain them between 0 and 1
            box_confidences.append(
                1 / (1 + np.exp(-output[..., 4, np.newaxis]))
            )

            # Applying sigmoid to box class scores
            #   to constrain them between 0 and 1
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))

        # return as tuple
        return boxes, box_confidences, box_class_probs
