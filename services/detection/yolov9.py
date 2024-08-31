import numpy as np
import torch
from loguru import logger
from opentelemetry import trace
from torchvision.transforms import Resize
from ultralytics import YOLO
from ultralytics.engine.results import Results as UltralyticsResults

YOLOV9_INPUT_SIZE_HW = (640, 640)
tracer = trace.get_tracer_provider().get_tracer(__name__)


class YoloV9Wrapper:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.detector_preproc = Resize(size=(640, 640))
        self.detector = YOLO("yolov9e")
        self.detector.to(self.device)
        self.detector.info()

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        with tracer.start_as_current_span("preprocess"):
            img_tensor_resized = self.preproc(img_tensor)
        with tracer.start_as_current_span("inference"):
            results = self.detector(img_tensor_resized, classes=[0], max_det=50, conf=0.25)
        with tracer.start_as_current_span("postprocess"):
            bboxes_xyxy = self.postproc(img_tensor, results)
        logger.info(f"{len(bboxes_xyxy)} people detected")
        return bboxes_xyxy

    def preproc(self, img_tensor: torch.Tensor) -> torch.Tensor:
        img_tensor_resized = self.detector_preproc(img_tensor)
        return img_tensor_resized

    def postproc(self, img_tensor: torch.Tensor, results: list[UltralyticsResults]) -> torch.Tensor:
        """Rescale back bboxes to the original image size."""
        input_img_size_hw = torch.tensor(img_tensor.shape[2:])
        scale_hw = input_img_size_hw / torch.tensor(YOLOV9_INPUT_SIZE_HW)
        bboxes_xyxy = results[0].boxes.xyxy.detach().cpu()
        bboxes_xyxy[:, [0, 2]] *= scale_hw[1]
        bboxes_xyxy[:, [1, 3]] *= scale_hw[0]
        bboxes_xyxy = bboxes_xyxy.to(torch.int64)
        return bboxes_xyxy
