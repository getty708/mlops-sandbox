import torch


def exclude_too_small_bbox(bbox_xyxy: torch.Tensor, min_area: int = 50 * 50) -> torch.Tensor:
    """Exclude too small bbox from the input bbox_xyxy.

    Args:
        bbox_xyxy (torch.Tensor): (NUM_OBJ, 4)
        min_area (int): Minimum size of the bbox.

    Returns:
        torch.Tensor: (NUM_OBJ, 4)
    """
    bbox_sizes = (bbox_xyxy[:, 2] - bbox_xyxy[:, 0]) * (bbox_xyxy[:, 3] - bbox_xyxy[:, 1])
    bbox_idx = bbox_sizes >= min_area
    return bbox_xyxy[bbox_idx]
