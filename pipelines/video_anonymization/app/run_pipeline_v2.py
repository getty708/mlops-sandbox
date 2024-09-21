"""Make anoymized video with body part segmentation.
"""

from email.policy import default
from pathlib import Path

import click
import cv2
import numpy as np
import torch
from loguru import logger
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from pipelines.video_anonymization.app.base import BasePipeline
from services.detection.detr.model import DetrWrapper
from services.detection.visualize import dwraw_bounding_boxes
from services.segmentation.sapience.wrapper import SapienceSegWrapper

MAX_FRAME_ID_IN_DEBUG_MODE = 10
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_OUTPUT_DIR = Path("./outputs/")
FULLHD_IMAGE_SIZE_HW = (1080, 1920)

SEGMENTATION_TARGET_BODY_PASRTS = (
    "Face_Neck",
    "Hair",
    "Lower_Lip",
    "Upper_Lip",
    "Lower_Teeth",
    "Upper_Teeth",
    "Tongue",
)

DEFAULT_OUTPUT_DIR = Path("./outputs/seg_base")

OTEL_CONNECTOR_ENDPOINT = "otel-collector:4317"


# ====================
# OpenTelemetry Setup
# ====================
resource = Resource.create({"service.name": "my-ml-pipeline"})

# == Trace ==
tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider=tracer_provider)
# Send traces to OtelCollector.
tracer_provider.add_span_processor(
    span_processor=BatchSpanProcessor(
        span_exporter=OTLPSpanExporter(endpoint=OTEL_CONNECTOR_ENDPOINT, insecure=True)
    )
)
tracer = trace.get_tracer_provider().get_tracer(__name__)

# == Metrics ==
reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint=OTEL_CONNECTOR_ENDPOINT, insecure=True),
    export_interval_millis=1000,
)
metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))

meter = metrics.get_meter_provider().get_meter(__name__)
people_counter = meter.create_up_down_counter(
    name="detected_people_count",
    description="Number of people detected in the frame",
    unit="1",
)


class PeopeleCounter:
    def __init__(self):
        self.current_peopel_count = 0

    def add(self, new_value):
        difference = new_value - self.current_peopel_count
        people_counter.add(difference)
        self.current_peopel_count = new_value


pcounter = PeopeleCounter()

# ===============
#  Drawing Utils
# ===============


def draw_face_anonymization_mask_with_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw a circle mask on the face of the person in the image.
    Center is a nose keypoint, radius is the distance between the nose and the right/left shoulder.

    Args:
        img (np.ndarray): Original image. (H, W, C)
        mask (np.ndarray): Segmentation mask (H, W, CLASS=7)

    """
    mask = np.clip(mask.sum(axis=2), 0, 1)
    img[mask == 1, :] = 0
    return img


# ==========
#  Pipeline
# ==========


class Pipeline(BasePipeline):
    detector: DetrWrapper
    segmentation: SapienceSegWrapper

    def __init__(self, *args, draw_model_outputs: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.draw_model_outputs = draw_model_outputs

    def init_pipeline_modules(self):
        self.detector = DetrWrapper(device=_DEVICE)
        self.segmentation = SapienceSegWrapper(SEGMENTATION_TARGET_BODY_PASRTS)
        self.segmentation.load_model()

    def process_single_frame(self, img: np.ndarray) -> np.ndarray:
        img_tensor_unscaled = (
            torch.from_numpy(img).permute(2, 0, 1).flip(0).unsqueeze(0).to(_DEVICE)
        )
        img_tensor_unscaled = img_tensor_unscaled.float()
        img_tensor = img_tensor_unscaled / 255.0

        # == Model Inference ==
        with tracer.start_as_current_span("person_detection"):
            bboxes_xyxy = self.detector(img_tensor)
            pcounter.add(len(bboxes_xyxy))
        with tracer.start_as_current_span("segmentation"):
            print(
                "C1-1",
                img_tensor_unscaled.size(),
                img_tensor_unscaled.min(),
                img_tensor_unscaled.max(),
                img_tensor_unscaled[0].size(),
                bboxes_xyxy.size(),
            )
            mask = self.segmentation(img_tensor_unscaled[0], bboxes_xyxy)
            print("C2-1", mask.sum())

        # == Visualize Results ==
        with tracer.start_as_current_span("visualization"):
            mask_ch_last = mask.detach().cpu().numpy().transpose(1, 2, 0)
            img = draw_face_anonymization_mask_with_mask(img, mask_ch_last)
            if self.draw_model_outputs:
                img = dwraw_bounding_boxes(img, bboxes_xyxy.numpy())
                # img = draw_keypoints(img, keypoints.numpy())
        return img


@click.command()
@click.option("-v", "--video-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    show_default=True,
    help=f"Process only the first {MAX_FRAME_ID_IN_DEBUG_MODE} frames",
)
def main(
    video_path: Path,
    output_dir: Path,
    debug: bool = False,
):
    pipeline = Pipeline(
        video_path=video_path,
        output_dir=output_dir,
        proc_tracer=tracer,
        debug=debug,
    )
    pipeline.init_pipeline_modules()
    pipeline.run()


if __name__ == "__main__":
    main()
