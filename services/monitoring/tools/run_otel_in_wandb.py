import time

import click
import numpy as np
import torch
import wandb
from loguru import logger
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
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
tracer_provider.add_span_processor(
    span_processor=BatchSpanProcessor(
        span_exporter=OTLPSpanExporter(endpoint="localhost:4317", insecure=True)
    )
)
tracer = trace.get_tracer_provider().get_tracer(__name__)


# == Metrics ==
reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint="localhost:4317", insecure=True),
    export_interval_millis=1000,
)
metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))

meter = metrics.get_meter_provider().get_meter(__name__)
people_counter = meter.create_counter(
    name="detected_people_count",
    description="Number of people detected in the frame",
    unit="1",
)


# ==============
#  Dummy Models
# ==============


class DummyModel(torch.nn.Module):
    def __init__(  # noqa: R0917
        self,
        amp: float = 10,  # milliseconds
        interval: float = 200,
        beta: float = 100,  # milliseconds
    ) -> None:
        super().__init__()
        self.sin_wave_fn = lambda x: amp * np.sin(2 * np.pi * x / interval) + beta
        self.call_count: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        duration = self.sin_wave_fn(self.call_count)  # milliseconds
        self.call_count += 1

        logger.info(f"Sleep for {duration:.2f}ms")
        time.sleep(duration / 1e3)
        return x


def num_detection_generator(frame_idx: int) -> int:
    """Generate the number of detection with a sine wave."""
    amp = 10  # amplitude = (the number of maximum detection) / 2
    interval = 1000  # the period of the sine wave
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
    model1 = DummyModel(amp=50, interval=250, beta=100)
    model2 = DummyModel(amp=50, interval=500, beta=100)
    model3 = DummyModel(amp=50, interval=1000, beta=100)

    # Run the pipeline
    for i in range(num_frames):
        logger.info(f"Processing frame {i}")
        with tracer.start_as_current_span("process_single_frame") as root_span:
            num_detection = num_detection_generator(i)
            x = torch.randn(1, 3, 100, 100)

            with tracer.start_as_current_span("model1"):
                x = model1(x)
            with tracer.start_as_current_span("model2"):
                x = model2(x)
            with tracer.start_as_current_span("model3"):
                x = model3(x)

            # Add metadata of this batch
            people_counter.add(num_detection)
            root_span.set_attribute("num_detecton", num_detection)
            root_span.set_attribute("frame_idx", i)
            # Mark this as root span and update the step in wandb.
            root_span.set_attribute(IS_ROOT_SPAN_KEY_NAME, True)

    trace.get_tracer_provider().shutdown()
    wandb.finish()


if __name__ == "__main__":
    main()
