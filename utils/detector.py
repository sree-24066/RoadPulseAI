from typing import NamedTuple
import numpy as np

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

def run_detection(net, image_array, conf_threshold):
    """Run YOLO detection on a numpy image array (RGB, any size)."""
    import cv2
    h_ori, w_ori = image_array.shape[:2]
    image_resized = cv2.resize(image_array, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=conf_threshold, verbose=False)

    from utils.model_loader import CLASSES
    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for _box in boxes:
            detections.append(Detection(
                class_id=int(_box.cls.item()),
                label=CLASSES[int(_box.cls.item())],
                score=float(_box.conf.item()),
                box=_box.xyxy[0].astype(int),
            ))

    annotated = results[0].plot()
    annotated_resized = cv2.resize(annotated, (w_ori, h_ori), interpolation=cv2.INTER_AREA)
    return detections, annotated_resized
