import torch
from services.toy_models.simple_transformer.const import (
    SIMPLE_TRANSFORMER_EMBEDDING_DIM,
    SIMPLE_TRANSFORMER_OUTPUT_TENSOR,
    SIMPLE_TRANSFORMER_SRC_SRQUENCE_LEN,
)
from services.toy_models.simple_transformer.simple_transformer import SimpleTransformer

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_smoke_simple_transformer():
    model = SimpleTransformer().to(_DEVICE)
    model.eval()
    assert isinstance(model, torch.nn.Module)

    batch_size = 32
    input_tensor = (
        torch.randn(
            (
                batch_size,
                SIMPLE_TRANSFORMER_SRC_SRQUENCE_LEN,
                SIMPLE_TRANSFORMER_EMBEDDING_DIM,
            )
        )
        .float()
        .to(_DEVICE)
    )
    output = model(input_tensor)

    assert isinstance(output, dict)
    out_tensor = output[SIMPLE_TRANSFORMER_OUTPUT_TENSOR]
    assert isinstance(out_tensor, torch.Tensor)
    assert list(out_tensor.size()) == [
        batch_size,
        SIMPLE_TRANSFORMER_SRC_SRQUENCE_LEN,
        SIMPLE_TRANSFORMER_EMBEDDING_DIM,
    ]
    if torch.cuda.is_available():
        assert out_tensor.is_cuda is True
    else:
        assert out_tensor.is_cuda is False
