import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from loguru import logger
from opentelemetry import trace

from services.segmentation.sapiens.const import SAPIENS_MODEL_DIR
from services.segmentation.sapiens.external.classes_and_palettes import GOLIATH_CLASSES

SAPIENS_INPUT_SIZE_HW = (1024, 768)


# == Otel ==
tracer = trace.get_tracer_provider().get_tracer(__name__)


# == Utils ==
def convert_model_outout_to_onehot_mask(
    model_output: torch.Tensor, output_heigh: int, output_width: int
) -> torch.Tensor:
    """Convert model output to onehot mask.

    Args:
        model_outputs: (NUM_CLASS, H, W)
    Returns:
        onehot_mask: (NUM_CLASS, H, W)
    """
    assert model_output.dim() == 3, f"model_output must be 3D tensor: {model_output.size()}"

    seg_logits = F.interpolate(
        model_output.unsqueeze(0), size=(output_heigh, output_width), mode="bilinear"
    ).squeeze(0)
    num_class = seg_logits.size(0)
    pred_sem_seg = seg_logits.argmax(dim=0, keepdim=True).detach().cpu()[0]
    onehot_mask = torch.eye(num_class)[pred_sem_seg].permute(2, 0, 1).to(dtype=torch.uint8)
    return onehot_mask


def merge_bbox_masks_on_original_image(
    model_outputs: torch.Tensor, bbox_xyxy: torch.Tensor, input_height: float, input_width: float
):
    """Merge model outputs for each bbox to the original image as an onehot mask format.

    Args:
        model_outputs: (NUM_CROPS, NUM_CLASS, Ho, Wo)
        bbox_xyxy: (NUM_CROPS, 4)
        input_height: Height of the original image
        input_width: Width of the original image
    Returns:
        onehot_mask_frame: (NUM_CLASS, H, W)
    """
    num_class = model_outputs.size(1)
    onehot_mask_frame = torch.zeros(size=(num_class, input_height, input_width), dtype=torch.uint8)

    # Set mask for each bbox to the frame mask in one-hot format.
    num_bbox = bbox_xyxy.size(0)
    for i in range(num_bbox):
        xyxy = bbox_xyxy[i]
        box_width = xyxy[2] - xyxy[0]
        box_height = xyxy[3] - xyxy[1]

        # Get segmentation mask for the bbox
        onehot_mask_bbox = convert_model_outout_to_onehot_mask(
            model_outputs[i], box_height, box_width
        )

        # Set the mask to the frame
        # TODO: Improve merging strategy of the overlapping area.
        onehot_mask_frame[:, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]] += onehot_mask_bbox

    return onehot_mask_frame


# == Wrapper ==
class SapiensSegWrapper:
    """Wrapper for Sapiens segmentation model.
    This model returns a segmentation mask for each in onehot-encording format.
    """

    model: torch.jit.ScriptModule

    def __init__(
        self,
        model_size: str = "0.6b",
        target_class_names: tuple[str] | None = None,
        device: str = "cuda",
    ):
        self.model_size = model_size
        self.device = device
        self.dtype = torch.float32

        # Params fror normalization are retrieved from the original demo script.
        # https://github.com/facebookresearch/sapiens/blob/c8d7a4106a6d82a4b33e2ca740b02c2f1bd9a366/lite/demo/vis_seg.py#L232-L233
        self.preproc_pipeline = T.Compose(
            [
                T.Resize(SAPIENS_INPUT_SIZE_HW),
                T.Normalize(mean=[123.5, 116.5, 103.5], std=[58.5, 57.0, 57.5]),
            ],
        )
        # Batch size for inference
        self.max_sub_batch_size = 8

        # Target Class
        if target_class_names is None:
            target_class_names = GOLIATH_CLASSES
        self.target_class_names = target_class_names
        self.target_class_idx = [GOLIATH_CLASSES.index(c) for c in target_class_names]
        logger.info(
            f"Target classes ({len(self.target_class_names)} classes): {self.target_class_names}"
        )

    def load_model(self):
        """Load model (TorchScript version)

        Ref: https://huggingface.co/collections/facebook/sapiens-66d22047daa6402d565cb2fc
        """
        checkpoint = list(
            SAPIENS_MODEL_DIR.glob(f"facebook/sapiens-seg-{self.model_size}-torchscript/*.pt2")
        )
        if len(checkpoint) == 0:
            raise RuntimeError(
                f"No checkpoints found for sapiens-seg-{self.model_size}-torchscript. Download weight in advance."
            )
        if len(checkpoint) > 1:
            raise RuntimeError(f"Multiple checkpoints found: {checkpoint}")
        checkpoint = checkpoint[0]
        logger.info(f"Loading model from {checkpoint}")

        model = torch.jit.load(checkpoint)
        self.dtype = torch.float32  # TorchScript models use float32
        self.model = model.to(self.device)

    def __call__(
        self, img_tensor: torch.Tensor, bboxes_xyxy: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Inference

        Args:
            img_tensor (torch.Tensor): (C, H, W)
            bboxes_xyxy (torch.Tensor): (NUM_CROPS, 4)
        """
        with tracer.start_as_current_span("preprocess"):
            cropped_imgs = self.preproc(img_tensor, bboxes_xyxy)
        with tracer.start_as_current_span("inference"):
            model_outputs = self.inference(cropped_imgs)
        with tracer.start_as_current_span("postprocess"):
            mask_frame = self.postproc(img_tensor, model_outputs, bboxes_xyxy)
        return mask_frame

    def preproc(
        self, img_tensor: torch.Tensor, bboxes_xyxy: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Crop bbox and apply preprocessing
        Args:
            img_tensor: (C, H, W), valuera in [0, 255].
            bboxes_xyxy: (NUM_CROPS, 4)
        Returns:
            cropped_imgs with shape=(NUM_CROPS, C, Hi=1024, Wi=768) or images with shape=(B=1, C, Hi, Wi)
        """
        img_tensor = img_tensor.to(self.device, dtype=self.dtype)

        # Feed the whole image to the model
        if bboxes_xyxy is None:
            img_tensor = self.preproc_pipeline(img_tensor)
            return img_tensor.unsqueeze(0)

        # Crop the image with bbox.
        # TODO: Refactor with torchvision.tv_tensors.BoundingBoxes
        cropped_imgs = []
        for xyxy in bboxes_xyxy:
            _cropped_img = img_tensor[:, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]  # .clone()
            _cropped_img = self.preproc_pipeline(_cropped_img)

            cropped_imgs.append(_cropped_img)
        cropped_imgs = torch.stack(cropped_imgs, dim=0).contiguous()
        return cropped_imgs

    def inference(self, cropped_imgs: torch.Tensor) -> torch.Tensor:
        """Run inference for the cropped images with batch processing.
        The input cropped images are split into small batches to avoid OOM error.

        Args:
            cropped_imgs: (NUM_CROPS or 1, C, Hi=1024, Wi=768)
        Returns:
            model_outputs: (NUM_CROPS, 512, 384)
        """
        model_outputs = []
        num_sub_batches = int(np.ceil(len(cropped_imgs) / self.max_sub_batch_size))
        for i in range(num_sub_batches):
            with torch.no_grad():
                sub_batch = cropped_imgs[
                    self.max_sub_batch_size * i : self.max_sub_batch_size * (i + 1)
                ]
                _model_outputs = self.model(sub_batch)
            model_outputs.append(_model_outputs)
        model_outputs = torch.cat(model_outputs, dim=0).detach().cpu().contiguous()
        return model_outputs

    def postproc(
        self,
        img_tensor: torch.Tensor,
        model_outputs: torch.Tensor,
        bbox_xyxy: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Return segmentation mask in the original image size. Segmentation mask for each bbox are
        projected to the original image.

        Args:
            img_tensor: an original input frame. (C, H, W)
            model_outputs: outputs from the sapiens. (NUM_CROPS, 512, 384)
            bbox_xyxy (torch.Tensor): bbox. (NUM_CROPS, 4)
        Returns:
            mask_frame (torch.Tensor): (NUM_CLASS, H, W)
        """
        input_height, input_width = img_tensor.size()[1:]

        if bbox_xyxy is None:
            seg_logits = model_outputs.squeeze(0)  # Drop dummy batch dim
            onehot_mask = convert_model_outout_to_onehot_mask(seg_logits, input_height, input_width)
        else:
            onehot_mask = merge_bbox_masks_on_original_image(
                model_outputs, bbox_xyxy, input_height, input_width
            )

        # Select results for target classes
        onehot_mask = torch.clamp(onehot_mask, 0, 1)[self.target_class_idx]
        return onehot_mask
