#!/usr/bin/env python3

if __name__ == "__main__":
    import numpy as np

    Yolo = __import__("6-yolo").Yolo

    np.random.seed(0)
    anchors = np.array(
        [
            [[116, 90], [156, 198], [373, 326]],
            [[30, 61], [62, 45], [59, 119]],
            [[10, 13], [16, 30], [33, 23]],
        ]
    )
    yolo = Yolo("yolo.h5", "coco_classes.txt", 0.6, 0.5, anchors)
    images, image_paths = yolo.load_images("./yolo")
    pimages, image_shapes = yolo.preprocess_images(images)
    predictions = []
    outputs = yolo.model.predict(pimages)
    for i, image_shape in enumerate(image_shapes):
        outs = [output[i] for output in outputs]
        boxes, box_confidences, box_class_probs = yolo.process_outputs(
            outs, image_shape
        )
        boxes, box_classes, box_scores = yolo.filter_boxes(
            boxes, box_confidences, box_class_probs
        )
        boxes, box_classes, box_scores = yolo.non_max_suppression(
            boxes, box_classes, box_scores
        )
        file_name = image_paths[i].split("/")[-1]
        yolo.show_boxes(images[i], boxes, box_classes, box_scores, file_name)
