"""The official GridWorld CNN state encoder and decoder for a grid of any size. Registered as "cnn_vae_any".

The lab's `cnn_vae` parts refuse every grid size but 10. These are the same layers, with the flattened size and the two
output paddings worked out from the grid size, so a 10x10 grid gets exactly the official shapes (10 -> 5 -> 2 and back) and
a 15x15 grid gets 15 -> 8 -> 3 and back. Nothing else differs.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from components.encoders import GaussianPosterior, embed_grid
from core.registry import DECODERS, ENCODERS


def conv_sizes(grid_size: int) -> tuple[int, int]:
    first = (grid_size + 2 - 3) // 2 + 1  # 3x3, stride 2, padding 1
    second = (first - 3) // 2 + 1  # 3x3, stride 2, padding 0
    if second < 1:
        raise ValueError(f"grid_size {grid_size} is too small for the two stride-2 convolutions")
    return first, second


class CNNVAEAnyEncoder(nn.Module):
    def __init__(self, grid_size: int, num_colors: int, state_dim: int = 32, num_state_tokens: int = 1, dropout: float = 0.0,
                 variational: bool = True, deterministic: bool = True) -> None:
        super().__init__()
        self.grid_size, self.num_colors, self.state_dim, self.num_state_tokens = grid_size, num_colors, state_dim, num_state_tokens
        self.variational, self.deterministic = variational, deterministic
        _, second = conv_sizes(grid_size)
        self.embedding = nn.Embedding(num_colors, 16)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(dropout),
            nn.Conv2d(128, 256, 3, stride=2, padding=0), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout2d(dropout),
        )
        flattened, output = 256 * second * second, num_state_tokens * state_dim
        self.to_state = nn.Linear(flattened, output)
        if variational:
            self.to_mean = nn.Linear(flattened, output)
            self.to_logvar = nn.Linear(flattened, output)

    def encode(self, grid: Tensor) -> tuple[Tensor, GaussianPosterior | None]:
        batch = grid.shape[0]
        features = self.conv_layers(embed_grid(grid, self.embedding).permute(0, 3, 1, 2)).reshape(batch, -1)
        shape = (batch, self.num_state_tokens, self.state_dim)
        if not self.variational:
            return F.normalize(self.to_state(features).view(shape), dim=-1), None
        posterior = GaussianPosterior(self.to_mean(features).view(shape), self.to_logvar(features).view(shape))
        return (posterior.mean if self.deterministic else posterior.sample()), posterior

    def forward(self, grid: Tensor) -> Tensor:
        return self.encode(grid)[0]


class CNNVAEAnyDecoder(nn.Module):
    def __init__(self, grid_size: int, num_colors: int, state_dim: int = 32, num_state_tokens: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        first, second = conv_sizes(grid_size)
        self.second = second
        self.from_state = nn.Linear(num_state_tokens * state_dim, 256 * second * second)
        pad_first = first - ((second - 1) * 2 + 3)
        pad_second = grid_size - ((first - 1) * 2 - 2 + 3)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0, output_padding=pad_first), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(dropout),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=pad_second), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(dropout),
            nn.Conv2d(64, num_colors, 3, padding=1),
        )

    def forward(self, state: Tensor) -> Tensor:
        batch = state.shape[0]
        features = self.from_state(state.reshape(batch, -1)).view(batch, 256, self.second, self.second)
        return self.deconv_layers(features).permute(0, 2, 3, 1)


ENCODERS["cnn_vae_any"] = CNNVAEAnyEncoder
DECODERS["cnn_vae_any"] = CNNVAEAnyDecoder
