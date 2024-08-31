import torch
import torch.nn as nn
import torch.nn.functional as F
from services.toy_models.simple_cnn.const import SIMPLE_CNN_OUTPUT_TENSOR


class SimpleCNN(nn.Module):
    def __init__(self, num_layers: int = 3):
        super().__init__()

        self.convs = []
        self.conv1 = nn.Conv2d(3, 3, (3, 3), stride=1, padding=(1, 1))
        self.convs.append(self.conv1)
        if num_layers >= 2:
            self.conv2 = nn.Conv2d(3, 3, (3, 3), stride=1, padding=(1, 1))
            self.convs.append(self.conv2)
        if num_layers >= 3:
            self.conv3 = nn.Conv2d(3, 3, (3, 3), stride=1, padding=(1, 1))
            self.convs.append(self.conv3)

    def forward(self, image_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        x = image_tensor
        for conv in self.convs:
            x = F.relu(conv(x))
        return {SIMPLE_CNN_OUTPUT_TENSOR: x}
