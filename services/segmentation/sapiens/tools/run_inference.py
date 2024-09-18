#!/usr/bin/env python
"""Run the sapeiens inference on a cropped image with SapiensSegWrapper and make a image with the predicted masks.
"""

from pathlib import Path

import click
import cv2
import numpy as np
import torch
from loguru import logger

from services.detection.detr.model import DetrWrapper
from services.detection.visualize import dwraw_bounding_boxes
from services.segmentation.sapiens.external.classes_and_palettes import GOLIATH_PALETTE
from services.segmentation.sapiens.wrapper import SapiensSegWrapper

_DEFAULT_OUTPUT_DIR = Path().cwd() / "outputs"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_img_tensor(img_path: Path) -> tuple[np.array, torch.Tensor]:
    """Load image and return it as a numpy with BGR/Channel-last format and a tensor with RGB/Channel-first format."""
    logger.info(f"Reading image from {img_path}")
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img).permute(2, 0, 1).float()
    return img, img_tensor


def onehot_semantic_mask_to_mask_bgr_img(
    mask_onehot: np.ndarray, target_class_idx: tuple[int]
) -> np.ndarray:
    """Convert onehot semantic mask to a mask image with BGR format.
    Color for each mask is the same as the sample in the hugging face.
    https://huggingface.co/spaces/facebook/sapiens-seg

    Args:
        mask_onehot: Onehot semantic mask with shape (H, W, NUM_CLASS).
        target_class_idx: Index of the target classes in GOLIATH_CLASSES.
    Returns:
        mask_img: Mask image with BGR format. (H, W, C)
    """
    assert mask_onehot.shape[2] == len(
        target_class_idx
    ), f"NUM_CLASS in mask_onehot be maching with len(target_class_idx): {target_class_idx}"

    img_height, img_width = mask_onehot.shape[:2]
    if 0 not in target_class_idx:
        # Add background class
        mask_onehot = np.concatenate(
            [
                np.zeros(shape=(img_height, img_width, 1)),
                mask_onehot,
            ],
            axis=0,
        )

    # Convert to 2D mask
    mask_with_bg = np.argmax(mask_onehot, axis=2).astype(np.uint8)
    # Convert 2D mask to RGB image
    mask_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for i, cls_idx in enumerate(target_class_idx):
        mask_img[mask_with_bg == i] = GOLIATH_PALETTE[cls_idx]

    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
    return mask_img


@click.command()
@click.option(
    "-i",
    "--input-img",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    required=True,
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=_DEFAULT_OUTPUT_DIR,
    show_default=True,
)
@click.option(
    "--sapiens-model-size",
    type=str,
    default="0.6b",
    show_default=True,
    help="parameter size of the sapiens.",
)
@click.option(
    "--apply-detector",
    is_flag=True,
    default=False,
    help="Apply detector and run sapiens on detected bboxes. If False, run sapiens on the whole image.",
)
@click.option(
    "--face-only",
    is_flag=True,
    help="Output masks only around the face region.",
)
@click.option(
    "--opacity",
    type=float,
    default=0.5,
    show_default=True,
    help="Opacity of the mask overlay.",
)
def main(
    input_img: Path,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    sapiens_model_size: str = "0.6b",
    apply_detector: bool = False,
    face_only: bool = False,
    opacity: float = 0.5,
):
    img, img_tensor = load_img_tensor(input_img)

    # == Init wrappers ==
    # Detector
    if apply_detector:
        detector = DetrWrapper(device=_DEVICE)
    # Sapiens
    if face_only:
        target_class_names = (
            "Face_Neck",
            "Hair",
            "Lower_Lip",
            "Upper_Lip",
            "Lower_Teeth",
            "Upper_Teeth",
            "Tongue",
        )
    else:
        target_class_names = None
    wrapper = SapiensSegWrapper(
        model_size=sapiens_model_size, target_class_names=target_class_names
    )
    wrapper.load_model()

    # == Inference ==
    if apply_detector:
        logger.info("Running detector ...")
        img_tensor_normalized = img_tensor / 255.0
        bboxes_xyxy = detector(img_tensor_normalized.unsqueeze(0))
    else:
        bboxes_xyxy = None
    logger.info("Running sapiens ...")
    mask_onehot = wrapper(img_tensor, bboxes_xyxy)

    # == Visualization ==
    # Onehot mask to BGR mask image
    mask_onehot_ch_last = mask_onehot.permute(1, 2, 0).numpy()
    mask_img = onehot_semantic_mask_to_mask_bgr_img(mask_onehot_ch_last, wrapper.target_class_idx)
    # Overlay mask on the input image
    vis_image = (img * (1 - opacity) + mask_img * opacity).astype(np.uint8)
    if apply_detector:
        vis_image = dwraw_bounding_boxes(vis_image, bboxes_xyxy.numpy())

    # Save mask
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_img.stem}_mask.png"
    cv2.imwrite(output_path, mask_img)
    logger.info(f"Saved {output_path}")
    # Save image with mask
    output_path = output_dir / f"{input_img.stem}_mask_overlay.png"
    cv2.imwrite(output_path, vis_image)
    logger.info(f"Saved {output_path}")


if __name__ == "__main__":
    main()
