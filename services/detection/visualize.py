import cv2
import numpy as np
import torch


def dwraw_bounding_boxes(img: np.ndarray, boxes: np.ndarray, thickness: int = 1) -> np.ndarray:
    for xyxy in boxes:
        img = cv2.rectangle(img, xyxy[:2], xyxy[2:], color=(0, 0, 255), thickness=thickness)
    return img
