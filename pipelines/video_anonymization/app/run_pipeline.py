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
from services.pose_estimation.transpose import TransPoseWrapper
from services.pose_estimation.visualize import draw_keypoints

MAX_FRAME_ID_IN_DEBUG_MODE = 10
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_OUTPUT_DIR = Path("./outputs/")
FULLHD_IMAGE_SIZE_HW = (1080, 1920)

_MSCOCO_NOSE_JOINT_INDEX = 0
_MSCOCO_LEFT_SHOULDER_JOINT_INDEX = 5
_MSCOCO_RIGHT_SHOULDER_JOINT_INDEX = 6

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


def draw_face_anonymization_mask(img: np.ndarray, keypoints_all: np.ndarray) -> np.ndarray:
    """Draw a circle mask on the face of the person in the image.
    Center is a nose keypoint, radius is the distance between the nose and the right/left shoulder.
    """
    for keypoints in keypoints_all:
        # Get face keypoints
        nose_keypoint = keypoints[_MSCOCO_NOSE_JOINT_INDEX]
        shoulder_keypoints = keypoints[
            [
                _MSCOCO_RIGHT_SHOULDER_JOINT_INDEX,
                _MSCOCO_LEFT_SHOULDER_JOINT_INDEX,
            ]
        ]
        radius = np.sqrt(np.sum((shoulder_keypoints - nose_keypoint) ** 2, axis=1)).max()

        # Draw mask
        img = cv2.circle(
            img,
            center=(int(nose_keypoint[0]), int(nose_keypoint[1])),
            radius=int(radius),
            color=(0, 0, 0),
            thickness=-1,
        )
    return img


# ==========
#  Pipeline
# ==========


class Pipeline(BasePipeline):
    detector: DetrWrapper
    pose_estimator: TransPoseWrapper

    def __init__(self, *args, draw_model_outputs: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.draw_model_outputs = draw_model_outputs

    def init_pipeline_modules(self):
        self.detector = DetrWrapper(device=_DEVICE)
        self.pose_estimator = TransPoseWrapper(device=_DEVICE)

    def process_single_frame(self, img: np.ndarray) -> np.ndarray:
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).flip(0).unsqueeze(0).to(_DEVICE)
        img_tensor = img_tensor.float() / 255.0

        # == Model Inference ==
        with tracer.start_as_current_span("person_detection"):
            bboxes_xyxy = self.detector(img_tensor)
            pcounter.add(len(bboxes_xyxy))
        with tracer.start_as_current_span("pose_estimation"):
            keypoints = self.pose_estimator(img_tensor, bboxes_xyxy)

        # == Visualize Results ==
        with tracer.start_as_current_span("visualization"):
            img = draw_face_anonymization_mask(img, keypoints.numpy())
            if self.draw_model_outputs:
                img = dwraw_bounding_boxes(img, bboxes_xyxy.numpy())
                img = draw_keypoints(img, keypoints.numpy())
        return img


@click.command()
@click.option("-v", "--video-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "-d",
    "--draw-model-outputs",
    is_flag=True,
    default=False,
    show_default=True,
    help="Draw keypoints on the image",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    show_default=True,
    help=f"Process only the first {MAX_FRAME_ID_IN_DEBUG_MODE} frames",
)
def main(video_path: Path, debug: bool = False, draw_model_outputs: bool = False):
    pipeline = Pipeline(
        video_path=video_path,
        proc_tracer=tracer,
        debug=debug,
    )
    pipeline.init_pipeline_modules()
    pipeline.run(draw_model_outputs=draw_model_outputs)


if __name__ == "__main__":
    main()
