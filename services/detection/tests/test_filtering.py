import numpy as np
import torch

from services.detection.filtering import exclude_too_small_bbox


def test_exclude_too_small_bbox():
    min_area = 50 * 50
    bbox_xyxy = torch.tensor(
        [
            [0, 0, 100, 200],  # valid bbox
            [0, 0, 20, 50],  # too small bbox
        ]
    )
    bbox_xyxy_expect = torch.tensor(
        [
            [0, 0, 100, 200],
        ]
    )

    bbox_xyxy_actual = exclude_too_small_bbox(bbox_xyxy, min_area=min_area)

    np.testing.assert_array_equal(bbox_xyxy_actual, bbox_xyxy_expect)
