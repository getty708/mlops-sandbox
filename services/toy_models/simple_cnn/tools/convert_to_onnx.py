from pathlib import Path

import click
import onnx
import torch
from loguru import logger
from services.toy_models.simple_cnn.const import (
    SIMPLE_CNN_INPUT_TENSOR,
    SIMPLE_CNN_OUTPUT_TENSOR,
)
from services.toy_models.simple_cnn.simple_cnn import SimpleCNN

DEFAULT_ONNX_MODEL_FILE_NAME = "simple_cnn.onnx"
DEFAULT_SIMPLE_CNN_ONNX_PATH = Path(__file__).parent / "outputs" / DEFAULT_ONNX_MODEL_FILE_NAME


def convert_pytorch_model_to_onnx(
    onnx_path: Path = DEFAULT_SIMPLE_CNN_ONNX_PATH,
    num_layers: int = 3,
):
    """Export SimpleCNN model to ONNX with Dynamo."""
    # Create a model instance and dummy outputs
    batch_size = 8
    model = SimpleCNN(num_layers=num_layers)
    dummy_input = torch.randn(batch_size, 3, 1080, 1920)

    # Convert to ONNX with Dynamo.
    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    onnx_program = torch.onnx.dynamo_export(model, dummy_input, export_options=export_options)
    onnx_program.save(str(onnx_path))


def rename_onnx_io_nodes(onnx_path: Path):
    """Rename input/output nodes in the ONNX model."""
    # Load the ONNX model
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    model = onnx.load(onnx_path)

    # Rename input/output node.
    original_input_node_name = model.graph.input[0].name
    original_output_node_name = model.graph.output[0].name
    logger.info(f"Rename input node: `{original_input_node_name}` to `{SIMPLE_CNN_INPUT_TENSOR}`.")
    model.graph.input[0].name = SIMPLE_CNN_INPUT_TENSOR
    logger.info(
        f"Rename output node: `{original_output_node_name}` to `{SIMPLE_CNN_OUTPUT_TENSOR}`."
    )
    model.graph.output[0].name = SIMPLE_CNN_OUTPUT_TENSOR

    # Rename the input/output name in the graph.
    logger.info(f"Rename input/output node name in the graph.")
    for node in model.graph.node:
        node.input[:] = [
            SIMPLE_CNN_INPUT_TENSOR if x == original_input_node_name else x for x in node.input
        ]
        node.output[:] = [
            SIMPLE_CNN_OUTPUT_TENSOR if x == original_output_node_name else x for x in node.output
        ]

    # Save the modified model
    onnx.save(model, onnx_path)


@click.command()
@click.option("-o", "--onnx-path", type=Path, default=DEFAULT_SIMPLE_CNN_ONNX_PATH)
@click.option("-l", "--num-layers", type=int, default=3)
def convert_simple_cnn_to_onnx(
    onnx_path: Path = DEFAULT_SIMPLE_CNN_ONNX_PATH,
    num_layers: int = 3,
):
    convert_pytorch_model_to_onnx(onnx_path, num_layers)
    rename_onnx_io_nodes(onnx_path)


if __name__ == "__main__":
    convert_simple_cnn_to_onnx()
