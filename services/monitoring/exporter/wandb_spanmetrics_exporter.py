from typing import Sequence

import wandb
from loguru import logger
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

IS_ROOT_SPAN_KEY_NAME = "is_root_span"


class WandBSpanmetricsExporter(SpanExporter):
    """Convert spans to metrics and log them with `wandb.log()`."""

    def __init__(self, commit_evry_call: bool = False) -> None:
        """_summary_

        Args:
            commit_evry_call (bool, optional): Set True when you have a only one span in every iteration.
                If not and you have multiple spans including nested spans, set False and set `is_root_span` attribute
                to spans that end in the last to increment 'step' in wandb. Defaults to False.
        """
        super().__init__()
        self.commit_evry_call = commit_evry_call

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            # Get duration from the span
            duration = (span.end_time - span.start_time) / 1e6  # Duration in milliseconds
            if self.commit_evry_call is True:
                commit = True
            else:
                commit = span.attributes.get(IS_ROOT_SPAN_KEY_NAME, False)
            metric = {f"otel/span/{span.name}/duration/ms": duration}
            # Get other attributes from the span
            for key, value in span.attributes.items():
                if key == IS_ROOT_SPAN_KEY_NAME:
                    continue
                metric[f"otel/span/{span.name}/{key}"] = value

            # Log the metric
            logger.debug("span: name={}, duration={:,}[ns]", span.name, duration)
            wandb.log(metric, commit=commit)

        # Return SUCCESS if successful
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        logger.info(f"Shutdown {self.__class__.__name__}")
