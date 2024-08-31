import cv2
import numpy as np
from services.pose_estimation.const import MSCOCO_SKELETON, NUM_JOINTS_MSCOCO


def draw_keypoints(img: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    for keypoints_single in keypoints:
        for i in range(0, NUM_JOINTS_MSCOCO):
            x = keypoints_single[i, 0]
            y = keypoints_single[i, 1]
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
        for p0_idx, p1_idx in MSCOCO_SKELETON:
            p0 = keypoints_single[p0_idx]
            p1 = keypoints_single[p1_idx]
            cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 1)

    return img
