import time

import pytest
import wandb
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from pytest_mock import MockerFixture

from services.monitoring.exporter.wandb_spanmetrics_exporter import (
    IS_ROOT_SPAN_KEY_NAME,
    WandBSpanmetricsExporter,
)


@pytest.fixture
def mocked_spans(mocker: MockerFixture) -> list[ReadableSpan]:
    num_spans = 5
    base_time = 0

    spans = []
    for i in range(num_spans):
        duration = i + 1.0

        # Child span
        span_child = mocker.MagicMock(spec=ReadableSpan)
        span_child.name = f"child-span-{i}-1"
        span_child.context = f"context-{i}"
        span_child.start_time = base_time
        span_child.end_time = base_time + 0.5
        span_child.attributes = {IS_ROOT_SPAN_KEY_NAME: True}
        span_child.status = "OK"
        spans.append(span_child)

        # Parent span
        span1 = mocker.MagicMock(spec=ReadableSpan)
        span1.name = f"test-span-{i}"
        span1.context = f"context-{i}"
        span1.start_time = base_time
        span1.end_time = base_time + duration
        span1.attributes = {IS_ROOT_SPAN_KEY_NAME: True}
        span1.status = "OK"
        spans.append(span1)

        base_time += span1.end_time

    return spans


def test_wandb_spanmetrics_exporter_export(mocker: MockerFixture, mocked_spans: list[ReadableSpan]):
    wandb.init(mode="disabled")
    mocke_wandb_log = mocker.patch(
        "services.monitoring.exporter.wandb_spanmetrics_exporter.wandb.log"
    )

    exporter = WandBSpanmetricsExporter()
    res = exporter.export(mocked_spans)

    assert res == SpanExportResult.SUCCESS
    assert mocke_wandb_log.call_count == len(mocked_spans)
    # Check the key starts with "otel/"
    args, _ = mocke_wandb_log.call_args
    assert list(args[0].keys())[0].startswith("otel/")


def test_wandb_spanmetrics_exporter(mocker: MockerFixture):
    wandb.init(mode="disabled")
    mocke_wandb_log = mocker.patch(
        "services.monitoring.exporter.wandb_spanmetrics_exporter.wandb.log"
    )
    # Init the tracer.
    resource = Resource.create({"service.name": "pipeline"})
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider=tracer_provider)
    tracer_provider.add_span_processor(
        span_processor=SimpleSpanProcessor(span_exporter=WandBSpanmetricsExporter())
    )
    tracer = trace.get_tracer_provider().get_tracer(__name__)

    # Generate spans
    num_spans = 5
    for i in range(num_spans):
        with tracer.start_as_current_span(f"test-span-{i}") as span:
            span.set_attribute(IS_ROOT_SPAN_KEY_NAME, True)
            time.sleep(0.1)

            with tracer.start_as_current_span(f"child-span-{i}") as span:
                time.sleep(0.1)

    # Check wandb.log() is called for 5 times.
    time.sleep(1.0)  # Wait for the exporter to finish.
    assert mocke_wandb_log.call_count == (num_spans * 2)
    # Check the number of wand.log() calls with commit=True.
    # The root span should have is_root_span=True to commit the results.
    num_commit_true = 0
    for _, kwargs in mocke_wandb_log.call_args_list:
        if kwargs.get("commit", False):
            num_commit_true += 1
    assert num_commit_true == num_spans
