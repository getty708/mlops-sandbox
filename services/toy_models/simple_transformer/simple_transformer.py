"""Ref: https://zenn.dev/turing_motors/articles/5b56edb7da1d30
"""

import torch
import torch.nn as nn
from services.toy_models.simple_transformer.const import (
    SIMPLE_TRANSFORMER_EMBEDDING_DIM,
    SIMPLE_TRANSFORMER_OUTPUT_TENSOR,
)


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = SIMPLE_TRANSFORMER_EMBEDDING_DIM,
        nhead: int = 2,
        num_layers: int = 2,
    ):
        super(SimpleTransformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers,
        )

    def forward(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        output_tensor = self.encoder(input_tensor)
        return {SIMPLE_TRANSFORMER_OUTPUT_TENSOR: output_tensor}
