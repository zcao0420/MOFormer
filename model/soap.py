import logging
import numpy as np
import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F

class regressoionHead(nn.Module):

    def __init__(self, d_embedding: int):
        super().__init__()
        self.layer1 = nn.Linear(d_embedding, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 1)
        self.relu=nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        
        return self.layer4(x)

