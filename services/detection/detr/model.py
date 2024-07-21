import torch
from loguru import logger
from opentelemetry import trace
from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput

tracer = trace.get_tracer_provider().get_tracer(__name__)

_DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_HUGGINGFACE_MODEL_NAME: str = "facebook/detr-resnet-50"
_HUGGINGFACE_MODEL_REVISION: str = "no_timm"

# outputs from the preprocessing step
_DETR_PIXEL_VALUES_TENSOR_NAME = "pixel_values"
# outputs from the model inference step
_DETR_LOGITS_TENSOR_NAME = "logits"
_DETR_PRED_BOXES_TENSOR_NAME = "pred_boxes"
# outputs from the postprocessing step
_DETR_SCORES_KEY_NAME = "scores"
_DETR_LABELS_KEY_NAME = "labels"
_DETR_BOXES_KEY_NAME = "boxes"


class DetrWrapper:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = DetrForObjectDetection.from_pretrained(_HUGGINGFACE_MODEL_NAME).to(
            device=_DEVICE
        )
        self.processor = DetrImageProcessor.from_pretrained(
            _HUGGINGFACE_MODEL_NAME, revision=_HUGGINGFACE_MODEL_REVISION
        )

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        with tracer.start_as_current_span("preprocess"):
            preproc_outputs = self.preproc(img_tensor)
        with tracer.start_as_current_span("inference"):
            model_outputs: dict[str, torch.Tensor] = self.model(
                pixel_values=preproc_outputs[_DETR_PIXEL_VALUES_TENSOR_NAME].to(device=self.device),
                return_dict=True,
            )
        with tracer.start_as_current_span("postprocess"):
            bboxes_xyxy = self.postproc(
                logits_batch=model_outputs[_DETR_LOGITS_TENSOR_NAME],
                pred_boxes_batch=model_outputs[_DETR_PRED_BOXES_TENSOR_NAME],
            )
        logger.info(f"{len(bboxes_xyxy)} people detected")
        return bboxes_xyxy

    def preproc(self, img_tensor: torch.Tensor) -> torch.Tensor:
        preproc_outputs: BatchFeature = self.processor(
            images=img_tensor,
            return_tensors="pt",
            data_format="channels_first",
            input_data_format="channels_first",
            do_rescale=False,
        )
        # Output from DetrImageProcessor is in CPU memory. Move them to GPU.
        preprocessed_img_tensor: dict[str, torch.Tensor] = preproc_outputs.data
        return preprocessed_img_tensor

    def postproc(
        self,
        logits_batch: torch.Tensor,
        pred_boxes_batch: torch.Tensor,
        frame_size: tuple[int, int] = (1080, 1920),
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Rescale back bboxes to the original image size."""
        batch_size: int = logits_batch.size(0)
        assert batch_size == 1
        detr_outputs = DetrObjectDetectionOutput(logits=logits_batch, pred_boxes=pred_boxes_batch)
        input_frame_shape = [frame_size] * batch_size
        outputs = self.processor.post_process_object_detection(
            detr_outputs, target_sizes=input_frame_shape, threshold=0.70
        )

        # Get the bounding boxes for the person class
        labels = outputs[0][_DETR_LABELS_KEY_NAME].to(dtype=torch.int64)
        bbox_idx_for_person = labels == 1
        boxes_xyxy_for_person = outputs[0][_DETR_BOXES_KEY_NAME][bbox_idx_for_person].to(
            dtype=torch.int64
        )
        return boxes_xyxy_for_person.detach().cpu()
