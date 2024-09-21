from pathlib import Path
from venv import logger

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from opentelemetry import trace

from services.segmentation.sapience.classes_and_palettes import GOLIATH_CLASSES

SAPIENCE_INPUT_SIZE_HW = (1024, 768)
SAPIENCE_ROOT_DIE = Path(__file__).parent

# == Otel ==
tracer = trace.get_tracer_provider().get_tracer(__name__)


class SapienceSegWrapper:
    model: torch.jit.ScriptModule

    def __init__(self, target_class_names: tuple[str] | None = None, device: str = "cuda"):
        self.device = device
        self.dtype = torch.float32
        self.preproc_pipeline = T.Compose(
            [
                T.Resize(SAPIENCE_INPUT_SIZE_HW),
                T.Normalize(mean=[123.5, 116.5, 103.5], std=[58.5, 57.0, 57.5]),
            ],
        )

        # Target Class
        if target_class_names is None:
            target_class_names = GOLIATH_CLASSES
        self.target_class_names = target_class_names
        self.target_class_idx = [GOLIATH_CLASSES.index(c) for c in target_class_names]
        logger.info(f"Target classes: {self.target_class_names}")

    def load_model(self):
        """Load model (TorchScript version)

        Ref: https://huggingface.co/facebook/sapiens-seg-0.3b-torchscript
        """
        model_size = "0.6b"
        filename = {
            "0.3b": "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2",
            "0.6b": "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2",
            "1b": "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
        }

        checkpoint = SAPIENCE_ROOT_DIE / "data" / "models" / filename[model_size]
        logger.info(f"Loading model from {checkpoint}")

        model = torch.jit.load(checkpoint)
        self.dtype = torch.float32  # TorchScript models use float32
        self.model = model.to(self.device)

    def __call__(self, img_tensor: torch.Tensor, bboxes_xyxy: torch.Tensor) -> torch.Tensor:
        """Inference

        Args:
            img_tensor (torch.Tensor): (C, H, W)
            bboxes_xyxy (torch.Tensor): (NUM_OBJ, 4)
        """
        with tracer.start_as_current_span("preprocess"):
            cropped_imgs, _ = self.preproc(img_tensor, bboxes_xyxy)
        with tracer.start_as_current_span("inference"):
            with torch.no_grad():
                model_outputs = self.model(cropped_imgs)
        with tracer.start_as_current_span("postprocess"):
            mask_frame = self.postproc(img_tensor, model_outputs, bboxes_xyxy)
        return mask_frame

    def preproc(
        self, img_tensor: torch.Tensor, bboxes_xyxy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Crop bbox and apply preprocessing
        Args:
            img_tensor (torch.Tensor): (C, H, W)
            bboxes_xyxy (torch.Tensor): (NUM_OBJ, 4)
        """
        img_tensor = img_tensor.to(self.device, dtype=self.dtype)

        # TODO: Refactor with torchvision.tv_tensors.BoundingBoxes
        cropped_imgs = []
        scales_xy = []
        for xyxy in bboxes_xyxy:
            _cropped_img = img_tensor[:, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]  # .clone()
            _cropped_img = self.preproc_pipeline(_cropped_img)

            cropped_imgs.append(_cropped_img)
            scales_xy.append(
                [
                    (xyxy[2] - xyxy[0]) / SAPIENCE_INPUT_SIZE_HW[1],
                    (xyxy[3] - xyxy[1]) / SAPIENCE_INPUT_SIZE_HW[0],
                ]
            )
        cropped_imgs = torch.stack(cropped_imgs, dim=0).contiguous()

        scales_xy = torch.tensor(scales_xy).contiguous()
        return cropped_imgs, scales_xy

    def postproc(
        self, img_tensor: torch.Tensor, result: torch.Tensor, bbox_xyxy: torch.Tensor
    ) -> list[torch.Tensor]:
        """Return segmentation mask in the original image size. Segmentation mask for each bbox are
        projected to the original image.

        Args:
            img_tensor (torch.Tensor): (C, H, W)
            result (torch.Tensor): model outputs, (NUM_OBJ, 512, 384)
            bbox_xyxy (torch.Tensor): bbox. (NUM_OBJ, 4)
        """
        classes = GOLIATH_CLASSES
        num_model_output_classes = len(classes)
        num_bbox = bbox_xyxy.size(0)

        # Prepare mask placeholder
        input_height, input_width = img_tensor.size()[1:]
        mask_frame = torch.zeros(
            size=(num_model_output_classes, input_height, input_width), dtype=torch.uint8
        )

        # Set mask for each bbox to the frame mask in one-hot format.
        for i in range(num_bbox):
            # box_width, box_height = bbox_wh[i]
            xyxy = bbox_xyxy[i]
            box_width = xyxy[2] - xyxy[0]
            box_height = xyxy[3] - xyxy[1]

            # Get segmentation mask for the bbox
            seg_logits = F.interpolate(
                result[[i]], size=(box_height, box_width), mode="bilinear"
            ).squeeze(0)
            pred_sem_seg = seg_logits.argmax(dim=0, keepdim=True)
            # Convert pred_sem_seg to onehot mask
            pred_sem_seg = pred_sem_seg.detach().cpu()[0]
            mask_bbox = (
                torch.eye(num_model_output_classes)[pred_sem_seg]
                .permute(2, 0, 1)
                .to(dtype=torch.uint8)
            )

            # Set the mask to the frame
            mask_frame[:, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]] += mask_bbox

        # Select results for target classes
        mask_frame = torch.clamp(mask_frame, 0, 1)[self.target_class_idx]
        return mask_frame
