from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from loguru import logger

from services.segmentation.sapiens.external.classes_and_palettes import (
    ORIGINAL_GOLIATH_PALETTE,
)
from services.segmentation.sapiens.wrapper import SapiensSegWrapper

_REPO_ROOT_DIR = Path(__file__).parents[4].resolve()


def get_sample_img_tensor() -> torch.Tensor:
    img_path = _REPO_ROOT_DIR / "data/pexels-anna-tarazevich-14751175-fullhd.jpg"
    logger.info(f"Reading image from {img_path}")
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img).permute(2, 0, 1).float()
    return img_tensor


def test_sapiens_seg_wrapper_preprocess():
    img_tensor = torch.randint(0, 255, (3, 1080, 1920)).float()
    bbox_xyxy = torch.tensor(
        [
            [0, 0, 100, 200],
            [500, 500, 600, 700],
        ]
    )

    wrapper = SapiensSegWrapper()
    cropped_imgs = wrapper.preproc(img_tensor, bbox_xyxy)

    # Check output tensor shapes
    np.testing.assert_array_equal(cropped_imgs.size(), (2, 3, 1024, 768))


def test_sapiens_seg_wrapper_preprocess_without_bbox():
    img_tensor = torch.randint(0, 255, (3, 1080, 1920)).float()

    wrapper = SapiensSegWrapper()
    cropped_imgs = wrapper.preproc(img_tensor)

    # Check output tensor shapes
    np.testing.assert_array_equal(cropped_imgs.size(), (1, 3, 1024, 768))


@pytest.mark.parametrize("use_bbox", [True, False])
def test_sapiens_seg_wrapper_postproc(use_bbox: bool):
    img_tensor = torch.randint(0, 255, (3, 1080, 1920)).float()
    if use_bbox:
        num_bbox = 2
        bbox_xyxy = torch.tensor(
            [
                [0, 0, 100, 200],
                [500, 500, 600, 700],
            ]
        )
        model_output = torch.rand(size=(num_bbox, 28, 512, 384)).float()
    else:
        num_bbox = 1
        bbox_xyxy = None
        model_output = torch.rand(size=(num_bbox, 28, 1080, 1920)).float()

    wrapper = SapiensSegWrapper()
    mask = wrapper.postproc(img_tensor, model_output, bbox_xyxy)

    # Check output tensor shapes
    np.testing.assert_array_equal(mask.size(), (28, 1080, 1920))


def test_sapiens_seg_wrapper_e2e():
    img_tensor = get_sample_img_tensor()
    bbox_xywh = torch.tensor(
        [
            [550, 30, 450, 800],
            [860, 250, 200, 450],
        ]
    )
    bbox_xyxy = torch.stack(
        [
            bbox_xywh[:, 0],
            bbox_xywh[:, 1],
            bbox_xywh[:, 0] + bbox_xywh[:, 2],
            bbox_xywh[:, 1] + bbox_xywh[:, 3],
        ],
        dim=1,
    )
    target_class_names = (
        "Face_Neck",
        "Hair",
        "Lower_Lip",
        "Upper_Lip",
        "Lower_Teeth",
        "Upper_Teeth",
        "Tongue",
    )

    wrapper = SapiensSegWrapper(target_class_names=target_class_names)
    wrapper.load_model()
    outputs = wrapper(img_tensor, bbox_xyxy)

    # == Save mask image ==
    # Add background class
    output_with_bg = np.concatenate([np.zeros(shape=(1, 1080, 1920)), outputs.numpy()], axis=0)
    mask_with_bg = np.argmax(output_with_bg, axis=0).astype(np.uint8)
    # Convert mask to RGB image
    mask_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    colors = ORIGINAL_GOLIATH_PALETTE[: len(target_class_names) + 1]
    for i, color in enumerate(colors):
        mask_img[mask_with_bg == i] = color
    # Save
    output_path = Path("./outputs/mask.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, mask_img)
    logger.info(f"Saved {output_path}")

    np.testing.assert_array_equal(outputs.size(), (len(target_class_names), 1080, 1920))
