"""Executor slot: f(state, code) -> state. Registered under EXECUTORS."""

from __future__ import annotations

from torch import Tensor, nn

from components.blocks.adaln import AdaLNZeroStack
from core.registry import EXECUTORS


class FiLMMLPExecutor(nn.Module):
    """Official NEO ProgramExecutor: FiLM-conditioned residual MLP on the flattened state."""

    def __init__(self, state_dim: int, action_dim: int, num_state_tokens: int, num_action_tokens: int, hidden: int, residual: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_state_tokens = num_state_tokens
        self.state_dim = state_dim
        self.residual = residual
        state_input_dim = num_state_tokens * state_dim
        self.state_layer = nn.Sequential(
            nn.Linear(state_input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.action_proj = nn.Linear(num_action_tokens * action_dim, hidden)
        self.film_gamma = nn.Linear(hidden, hidden)
        self.film_beta = nn.Linear(hidden, hidden)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, state_input_dim),
        )
        nn.init.ones_(self.film_gamma.weight.data.diagonal())
        nn.init.zeros_(self.film_gamma.bias.data)
        nn.init.zeros_(self.film_beta.weight.data)
        nn.init.zeros_(self.film_beta.bias.data)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        batch_size = state.shape[0]
        hidden = self.state_layer(state.reshape(batch_size, -1))
        action_embedding = self.action_proj(action.reshape(batch_size, -1))
        hidden = self.film_gamma(action_embedding) * hidden + self.film_beta(action_embedding)
        delta = self.output_layers(hidden).view(batch_size, self.num_state_tokens, self.state_dim)
        return state + delta if self.residual else delta


class AdaLNTransformer(nn.Module):
    """BabyTheorist AdaLNTransition: AdaLN-Zero blocks over state tokens, conditioned on the flattened code.
    grid_dim: state tokens are grid cells (token_embed, no positions) and attention uses 2D RoPE."""

    def __init__(self, state_dim: int, cond_dim: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float = 0.0, grid_dim: int | None = None) -> None:
        super().__init__()
        self.stack = AdaLNZeroStack(state_dim, cond_dim, d_model, num_layers, num_heads, d_ff, dropout, grid_dim)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.stack(state, action)


EXECUTORS["film_mlp"] = FiLMMLPExecutor
EXECUTORS["adaln_transformer"] = AdaLNTransformer
