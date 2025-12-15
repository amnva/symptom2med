import torch.nn as nn
import torch

class MedicineRecommender(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512), nn.GELU(), nn.Dropout(0.3), nn.BatchNorm1d(512),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x).squeeze()
