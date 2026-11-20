"""Programmer slot: infer code pre-activations. Registered under PROGRAMMERS."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from components.blocks.rope import RoPEEncoder
from components.blocks.transformer import init_embedding_scaled, init_linear_scaled
from core.registry import PROGRAMMERS


class FiLMMLPProgrammer(nn.Module):
    """Official NEO TheoryProgrammer: stepwise q(z_k | s_k, s_y). Length FiLM is unused by the paper rollout."""

    def __init__(self, state_dim: int, action_dim: int, num_state_tokens: int, num_action_tokens: int, hidden: int, max_length: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_action_tokens = num_action_tokens
        self.action_dim = action_dim
        input_dim = 2 * num_state_tokens * state_dim
        output_dim = num_action_tokens * action_dim
        self.length_embedding = nn.Embedding(max_length + 1, hidden)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.film_gamma = nn.Linear(hidden, hidden)
        self.film_beta = nn.Linear(hidden, hidden)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )
        nn.init.ones_(self.film_gamma.weight.data.diagonal())
        nn.init.zeros_(self.film_gamma.bias.data)
        nn.init.zeros_(self.film_beta.weight.data)
        nn.init.zeros_(self.film_beta.bias.data)

    def forward(self, current_state: Tensor, target_state: Tensor, length: Tensor | None = None) -> Tensor:
        batch_size = current_state.shape[0]
        hidden = self.input_layer(torch.cat([current_state.reshape(batch_size, -1), target_state.reshape(batch_size, -1)], dim=-1))
        if length is not None:
            length_embedding = self.length_embedding(length)
            hidden = self.film_gamma(length_embedding) * hidden + self.film_beta(length_embedding)
        return self.output_layers(hidden).view(batch_size, self.num_action_tokens, self.action_dim)


class TokenQueryTransformer(nn.Module):
    """Stepwise NEO programmer over cell-token states: interleaved (s_k, s_y) cell tokens (+ type) ++ query -> 2D-RoPE Transformer -> (B,1,action_dim).
    The trailing query token is not rotated."""

    def __init__(self, d_model: int, action_dim: int, num_heads: int, d_ff: int, num_layers: int, grid_dim: int, dropout: float = 0.0, layer_norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(2, d_model)
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        self.transformer = RoPEEncoder(d_model, num_layers, num_heads, d_ff, dropout, grid_dim, tokens_per_cell=2)
        self.output_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.output = nn.Linear(d_model, action_dim, bias=False)
        self.apply(init_linear_scaled)
        init_embedding_scaled(self.type_embedding)
        nn.init.normal_(self.query, std=d_model**-0.5)

    def forward(self, current_state: Tensor, target_state: Tensor) -> Tensor:
        """current_state, target_state (B,L,d) -> (B,1,action_dim); tokens s_1 y_1 s_2 y_2 ... then the query."""

        batch_size = current_state.shape[0]
        tokens = torch.stack((current_state, target_state), dim=2) + self.type_embedding.weight
        encoded = self.transformer(torch.cat((tokens.flatten(1, 2), self.query.expand(batch_size, -1, -1)), dim=1))
        return self.output(self.output_norm(encoded[:, -1]))[:, None, :]


PROGRAMMERS["film_mlp"] = FiLMMLPProgrammer
PROGRAMMERS["token_query_transformer"] = TokenQueryTransformer
