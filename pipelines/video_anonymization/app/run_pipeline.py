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
        span_exporter=OTLPSpanExporter(endpoint="otel-collector:4317", insecure=True)
    )
)
tracer = trace.get_tracer_provider().get_tracer(__name__)

# == Metrics ==
reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint="otel-collector:4317", insecure=True),
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


class Pipeline:
    def __init__(
        self,
        video_path: Path,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        frame_size_hw: tuple[int, int] = FULLHD_IMAGE_SIZE_HW,
        debug: bool = False,
    ):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_size_hw = frame_size_hw
        self.debug = debug
        self.people_counter = PeopeleCounter()

        # Init videos
        self.cap = cv2.VideoCapture(str(video_path))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        logger.info(
            f"Start pipeline with video: {video_path} "
            f"(frame count = {self.frame_count}, frame rate = {self.frame_rate})"
        )

        # Outputs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_frame_dir = output_dir / "frames"
        self.output_frame_dir.mkdir(parents=True, exist_ok=True)

        self.output_video_path = output_dir / f"{video_path.stem}_anonymized.mp4"
        logger.info(f"Output video path: {self.output_video_path}")

        # Init models
        self.detector = DetrWrapper(device=_DEVICE)
        self.pose_estimator = TransPoseWrapper(device=_DEVICE)

    def run(self, draw_model_outputs: bool = False):
        # Init video writer
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(
            str(self.output_video_path),
            fourcc,
            self.frame_rate,
            (self.frame_size_hw[1], self.frame_size_hw[0]),
        )

        # Process frames
        frame_id = 0
        while True:
            frame_id += 1
            try:
                with tracer.start_as_current_span("process_single_frame"):
                    # Get next frame
                    is_success, frame = self.cap.read()
                    if not is_success:
                        break
                    logger.info(f"Process frame {frame_id}/{self.frame_count}")

                    # Run inferences
                    frame = self.process_single_frame(frame, draw_model_outputs=draw_model_outputs)

                    # Save Images
                    video_writer.write(frame)
                    if self.debug:
                        img_path = self.output_frame_dir / f"img_{frame_id}.jpg"
                        cv2.imwrite(str(img_path), frame)

                    if self.debug and (frame_id >= MAX_FRAME_ID_IN_DEBUG_MODE):
                        logger.info(
                            f"Reached max frame id in debug mode: {MAX_FRAME_ID_IN_DEBUG_MODE}"
                        )
                        break
            except KeyboardInterrupt:
                break

        logger.info("Stop pipeline ...")
        video_writer.release()

    def process_single_frame(self, img: np.ndarray, draw_model_outputs: bool = False) -> np.ndarray:
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).flip(0).unsqueeze(0).to(_DEVICE)
        img_tensor = img_tensor.float() / 255.0

        # == Model Inference ==
        with tracer.start_as_current_span("person_detection"):
            bboxes_xyxy = self.detector(img_tensor)
            self.people_counter.add(len(bboxes_xyxy))
        with tracer.start_as_current_span("pose_estimation"):
            keypoints = self.pose_estimator(img_tensor, bboxes_xyxy)

        # == Visualize Results ==
        with tracer.start_as_current_span("visualization"):
            img = draw_face_anonymization_mask(img, keypoints.numpy())
            if draw_model_outputs:
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
        debug=debug,
    )
    pipeline.run(draw_model_outputs=draw_model_outputs)


if __name__ == "__main__":
    main()
