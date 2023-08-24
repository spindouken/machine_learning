#!/usr/bin/env python3

if __name__ == "__main__":
    import numpy as np

    Yolo = __import__("7-yolo").Yolo

    np.random.seed(0)
    anchors = np.array(
        [
            [[116, 90], [156, 198], [373, 326]],
            [[30, 61], [62, 45], [59, 119]],
            [[10, 13], [16, 30], [33, 23]],
        ]
    )
    yolo = Yolo("yolo.h5", "coco_classes.txt", 0.6, 0.5, anchors)
    predictions, image_paths = yolo.predict("./yolo")
    imgs = zip(predictions, image_paths)
    imgs = sorted(imgs, key=lambda v: v[1])
    predictions = [a for a, b in imgs]
    image_paths = [b for a, b in imgs]
    with open("0-test", "w+") as f:
        f.write(str(predictions))
    with open("1-test", "w+") as f:
        f.write(str(image_paths))
