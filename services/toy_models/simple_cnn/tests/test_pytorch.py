from unittest.mock import MagicMock, patch

import pytest
import torch
from services.toy_models.simple_cnn.const import SIMPLE_CNN_OUTPUT_TENSOR
from services.toy_models.simple_cnn.simple_cnn import SimpleCNN


@patch("torch.nn.Conv2d.forward")
@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_forward_conv2d_calls(mock_conv2d: MagicMock, num_layers: int):
    model = SimpleCNN(num_layers=num_layers)
    input_data = torch.randn(1, 3, 32, 32)
    mock_conv2d.return_value = torch.randn(1, 10, 32, 32)  # Set the return value for mock_conv2d

    output = model(input_data)

    # Check all layers are called.
    assert mock_conv2d.call_count == num_layers
    assert isinstance(output, dict)
    assert SIMPLE_CNN_OUTPUT_TENSOR in output
