from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from opentelemetry import trace

MAX_FRAME_ID_IN_DEBUG_MODE = 10

DEFAULT_OUTPUT_DIR = Path("./outputs/")
FULLHD_IMAGE_SIZE_HW = (1080, 1920)


class BasePipeline:
    def __init__(
        self,
        video_path: Path,
        proc_tracer: trace.Tracer,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        frame_size_hw: tuple[int, int] = FULLHD_IMAGE_SIZE_HW,
        debug: bool = False,
    ):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_size_hw = frame_size_hw
        self.debug = debug

        # Monitoring
        self.proc_tracer = proc_tracer

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

    def init_pipeline_modules(self):
        raise NotImplementedError

    def run(self):
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
                with self.proc_tracer.start_as_current_span("process_single_frame"):
                    # Get next frame
                    is_success, frame = self.cap.read()
                    if not is_success:
                        break
                    logger.info(f"Process frame {frame_id}/{self.frame_count}")

                    # Run inferences
                    frame = self.process_single_frame(frame)

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

    def process_single_frame(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError
