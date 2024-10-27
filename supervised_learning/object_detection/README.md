# Object Detection

## Colab page
https://colab.research.google.com/github/spindouken/holbertonschool-machine_learning/blob/master/supervised_learning/object_detection/YOLO.ipynb#scrollTo=75Ze0gzS9NaM

## Project Summary

This project implements the YOLO v3 (You Only Look Once version 3) algorithm for real-time object detection through a structured `Yolo` class. It encompasses various functionalities necessary for effective object detection, including model initialization, output processing, box filtering, and non-max suppression. Additionally, the project includes methods for image loading, preprocessing, visualizing detected boxes, and predicting objects in images.

### Key Features
- **Output Processing**
- **Bounding Box Filtering**
- **Non-max Suppression**
- **Preprocessing Images (Resize and Rescale)**: The project facilitates loading images from a specified folder into numpy arrays, followed by preprocessing methods that resize and rescale images to match the model's input requirements.
- **Bounding Box Visualization**

### YOLO Task Summaries
0. **Initialize Yolo**: Creates a `Yolo` class that initializes the model with paths to the Darknet Keras model and class names, along with parameters for box score threshold, IOU threshold for non-max suppression, and anchor boxes.
1. **Process Outputs**: Adds a method to the `Yolo` class to process model outputs, converting predictions into usable boundary boxes, confidences, and class probabilities relative to the original image size.
2. **Filter Boxes**: Extends the `Yolo` class with a method to filter bounding boxes based on confidence scores and class probabilities, returning filtered boxes, their classes, and scores.
3. **Non-max Suppression**: Implements non-max suppression in the `Yolo` class to refine detected bounding boxes by removing overlapping boxes based on their scores, returning the best predictions.
4. **Load Images**: Introduces a static method to the `Yolo` class that loads images from a specified folder into a list of numpy arrays, along with their file paths.
5. **Preprocess Images**: Adds a method to preprocess images by resizing and rescaling them, preparing them for the model input, and returning the processed images along with their original dimensions.
6. **Show Boxes**: Implements a method in the `Yolo` class to visualize detected bounding boxes on images, displaying class names and scores, with options to save the annotated images.
7. **Predict**: Enhances the `Yolo` class with a method to predict objects in images located in a specified folder, displaying the images with detected boxes and returning the predictions and paths.
