import time

import click
import numpy as np
import torch
import wandb
from loguru import logger
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from services.monitoring.exporter.wandb_spanmetrics_exporter import (
    IS_ROOT_SPAN_KEY_NAME,
    WandBSpanmetricsExporter,
)

WANDB_PROJECT_NAME = "otel-in-wandb"

# ====================
# OpenTelemetry Setup
# ====================
resource = Resource.create({"service.name": "pipeline"})

# == Trace ==
tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider=tracer_provider)
# Send traces to WandB as a metric
tracer_provider.add_span_processor(
    span_processor=BatchSpanProcessor(span_exporter=WandBSpanmetricsExporter())
)
tracer = trace.get_tracer_provider().get_tracer(__name__)

# ==============
#  Dummy Models
# ==============


class DummyModel(torch.nn.Module):
    def __init__(self, name: str, mean: float, std: float, seed: int = 12345) -> None:
        super().__init__()
        self.name = name
        self.mean = mean
        self.std = std
        self.rng = np.random.default_rng(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        duration = self.rng.normal(loc=self.mean, scale=self.std)
        logger.info(f"Sleep for {duration:.2f}sec")
        time.sleep(duration)
        return x


def num_detection_generator(frame_idx: int) -> int:
    """Generate the number of detection with a sine wave."""
    amp = 10  # amplitude = (the number of maximum detection) / 2
    interval = 20  # the period of the sine wave
    return int(amp * np.sin(2 * np.pi * frame_idx / interval) + amp)


# ======
#  Main
# ======


@click.command()
@click.option("-n", "--num-frames", default=10, help="Number of frames to process")
@click.option("-w", "--wandb-mode", default="offline", show_default=True, help="WandB mode")
def main(num_frames: int = 10, wandb_mode: str = "offline"):
    wandb.init(
        project=WANDB_PROJECT_NAME,
        mode=wandb_mode,
    )

    # Init dummy models
    model1 = DummyModel(name="model1", mean=0.1, std=0.001)
    model2 = DummyModel(name="model2", mean=0.2, std=0.010)
    model3 = DummyModel(name="model3", mean=0.3, std=0.100)

    # Run the pipeline
    for i in range(num_frames):
        logger.info(f"Processing frame {i}")
        with tracer.start_as_current_span("process_single_frame") as root_span:
            x = torch.randn(1, 3, 100, 100)
            with tracer.start_as_current_span("model1"):
                x = model1(x)

            with tracer.start_as_current_span("model2"):
                x = model2(x)
            with tracer.start_as_current_span("model3"):
                x = model3(x)

            # Add metadata of this batch
            root_span.set_attribute("num_detecton", num_detection_generator(i))
            root_span.set_attribute("frame_idx", i)
            # Mark this as root span and update the step in wandb.
            root_span.set_attribute(IS_ROOT_SPAN_KEY_NAME, True)

    trace.get_tracer_provider().shutdown()
    wandb.finish()


if __name__ == "__main__":
    main()
