#!/usr/bin/env python3
"""
uses the Yolo v3 algorithm to perform object detection
THIS CODE HAS NOT BEEN CHECKED FOR CORRECTNESS
WORKING VERSION IS IN YOLO.ipynb google colab page (with many edits)
"""
import tensorflow as tf
import numpy as np
import os
import cv2


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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        boxes: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 4)
            containing processed boundary boxes for each output, respectively
        box_confidences: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1)
            containing processed box confidences for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes)
            containing processed box class probabilities for each output

        Returns a tuple of (filtered_boxes, box_classes, box_scores):
            filtered_boxes: a numpy.ndarray of shape (?, 4)
                containing all of the filtered bounding boxes
            box_classes: a numpy.ndarray of shape (?,)
                containing the class number that each box
                in filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?)
                containing the box scores for each box
                in filtered_boxes, respectively
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        # Iterate through each output
        for box, box_confidence, box_class_prob in zip(
            boxes, box_confidences, box_class_probs
        ):
            # Compute box scores by multiplying confidence w/ class probs
            scores = box_confidence * box_class_prob

            # Get the class index with the maximum score for each box
            box_class_indices = np.argmax(scores, axis=-1)
            box_class_scores = np.max(scores, axis=-1)

            # Create a mask for scores greater than the threshold
            filtering_mask = box_class_scores >= self.class_t

            # Apply the mask to filter the boxes, class indices, and scores
            filtered_boxes.append(box[filtering_mask])
            box_classes.append(box_class_indices[filtering_mask])
            box_scores.append(box_class_scores[filtering_mask])

        # Concatenate results from all outputs into single arrays
        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        filtered_boxes: a numpy.ndarray of shape (?, 4)
            containing all of the filtered bounding boxes
        box_classes: a numpy.ndarray of shape (?,)
            containing the class number for the class that
            filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?)
            containing the box scores for each box in filtered_boxes

        Returns tuple
        (box_predictions, predicted_box_classes, predicted_box_scores):
            box_predictions: a numpy.ndarray of shape (?, 4)
            containing predicted bounding boxes ordered by class and box score
            predicted_box_classes: a numpy.ndarray of shape (?,)
                containing the class number for box_predictions
                ordered by class and box score, respectively
            predicted_box_scores: a numpy.ndarray of shape (?)
                containing the box scores for box_predictions
                ordered by class and box score, respectively
        """
        # Concatenate box classes and box scores to sort them together
        box_classes = box_classes.astype(int).reshape(-1, 1)
        box_scores = box_scores.reshape(-1, 1)
        boxes_with_classes_and_scores = np.concatenate(
            (filtered_boxes, box_classes, box_scores), axis=1
        )

        # Sort boxes by class and scores
        sortedBoxes = boxes_with_classes_and_scores[
            np.lexsort((box_scores.ravel(), box_classes.ravel()))
        ]

        # Pre-allocate result arrays
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Iterate through unique classes
        unique_classes = np.unique(sortedBoxes[:, -2])
        for unique_class in unique_classes:
            # Get boxes of the same class
            same_class_boxes = sortedBoxes[sortedBoxes[:, -2] == unique_class]

            # Apply non-max suppression for the same class boxes
            while len(same_class_boxes) > 0:
                # Take the box with the highest score
                best_box = same_class_boxes[-1]
                box_predictions.append(best_box[:4])
                predicted_box_classes.append(best_box[4])
                predicted_box_scores.append(best_box[5])

                # Remove the best box
                same_class_boxes = same_class_boxes[:-1]

                # Compute IoU with the best box using vectorized operations
                x1 = np.maximum(same_class_boxes[:, 0], best_box[0])
                y1 = np.maximum(same_class_boxes[:, 1], best_box[1])
                x2 = np.minimum(same_class_boxes[:, 2], best_box[2])
                y2 = np.minimum(same_class_boxes[:, 3], best_box[3])

                intersection_area = np.maximum(0, x2 - x1) * np.maximum(
                    0, y2 - y1
                )
                best_box_area = (best_box[2] - best_box[0]) * (
                    best_box[3] - best_box[1]
                )
                current_boxes_area = (
                    same_class_boxes[:, 2] - same_class_boxes[:, 0]
                ) * (same_class_boxes[:, 3] - same_class_boxes[:, 1])
                union_area = (
                    best_box_area + current_boxes_area - intersection_area
                )

                iou = intersection_area / union_area

                # Remove boxes with IoU greater than the threshold
                same_class_boxes = same_class_boxes[iou < self.nms_t]

        predicted_box_classes = np.array(predicted_box_classes).astype(int)
        return (
            np.array(box_predictions).astype(int),
            predicted_box_classes,
            np.array(predicted_box_scores),
        )

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a given folder path

        folder_path: a string representing the path to the
            folder holding all the images to load

        Returns a tuple of (images, image_paths):
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual images in images
        """
        images = []
        image_paths = []
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            if os.path.isfile(path):
                image = cv2.imread(path)
                if image is not None:
                    images.append(image)
                    image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Resizes and rescales the images before processing.

        images: a list of images as numpy.ndarrays
        Resize the images with inter-cubic interpolation
        Rescale all images to have pixel values in the range [0, 1]
        Returns a tuple of (pimages, image_shapes):
            pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
                containing all of the preprocessed images
                ni: the number of images that were preprocessed
                input_h: the input height for the Darknet model
                input_w: the input width for the Darknet model
                3: number of color channels
            image_shapes: a numpy.ndarray of shape (ni, 2)
                containing the original height and width of the images
                2 => (image_height, image_width)
        """
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]
        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append([image.shape[0], image.shape[1]])
            resized = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC,
            )
            rescaled = resized / 255
            pimages.append(rescaled)

        return (np.array(pimages), np.array(image_shapes))

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        image: a numpy.ndarray containing an unprocessed image
        boxes: a numpy.ndarray containing the boundary boxes for the image
        box_classes: a numpy.ndarray containing the class indices for each box
        box_scores: a numpy.ndarray containing the box scores for each box
        file_name: the file path where the original image is stored
        """
        # Iterate through the boxes and draw them on the image
        for i, box in enumerate(boxes):
            # Extract the coordinates
            x1, y1, x2, y2 = map(int, box)
            # Draw the rectangle (box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Prepare the text including class name and score
            class_name = self.class_names[box_classes[i]]
            score = round(box_scores[i], 2)
            label = "{} {:.2f}".format(class_name, score)
            # Put the text on the image
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # Show the image
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        # If 's' key is pressed, save the image
        if key == ord("s"):
            # Check if detections directory exists, if not, create it
            if not os.path.exists("detections"):
                os.makedirs("detections")
            # Save the image
            cv2.imwrite(os.path.join("detections", file_name), image)

        # Close the image window
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        makes detection predictions on all images
            in specified folder

        folder_path: a string representing the path
          to the folder holding all the images to predict
        All image windows should be named after the
          corresponding image filename without its full path
        Displays all images using the show_boxes method
        Returns: a tuple of (predictions, image_paths):
          predictions: a list of tuples for each image of
            (boxes, box_classes, box_scores)
          image_paths: a list of image paths corresponding
            to each prediction in predictions
        """
        # Load and preprocess images
        # 'images' contains the original images,
        #   'paths' contains the file paths
        # 'pimages' contains preprocessed images,
        #   'img_shapes' contains the original shapes
        images, paths = self.load_images(folder_path)
        pimages, img_shapes = self.preprocess_images(images)

        # Stack processed images into a single numpy array
        # This is done to create a batch that
        #   can be fed into the model for prediction
        img_stack = np.concatenate(
            [img[np.newaxis, ...] for img in pimages], axis=0
        )

        # Make a pass through the ConvNet to get raw predictions
        # The model's predict method returns raw predictions
        #   that need further processing
        raw_predictions = self.model.predict(img_stack)

        # Initialize lists to store final predictions and detection paths
        predictions = []
        detection_paths = []

        # Iterate through each image and its corresponding raw prediction
        for x, img in enumerate(images):
            # Extract the raw predictions for the current image
            # Raw predictions are divided into three parts,
            #   hence the loop through range(3)
            one_raw = [raw_predictions[i][x, ...] for i in range(3)]

            # Process the raw predictions to get boxes,
            #   confidences, and class probabilities
            boxes, confidences, class_probs = self.process_outputs(
                one_raw, img_shapes[x]
            )

            # Filter the boxes based on confidence and class probabilities
            boxes, box_classes, box_scores = self.filter_boxes(
                boxes, confidences, class_probs
            )

            # Apply non-maximum suppression to remove overlapping boxes
            box_preds, class_preds, score_preds = self.non_max_suppression(
                boxes, box_classes, box_scores
            )

            # Show the boxes on the image & append the path to detection_paths
            detection_paths.append(
                self.show_boxes(
                    img,
                    box_preds,
                    class_preds,
                    score_preds,
                    paths[x].rpartition("/")[-1],
                )
            )

            # Append the final predictions for the current image
            predictions.append((box_preds, class_preds, score_preds))

        return (predictions, detection_paths)
