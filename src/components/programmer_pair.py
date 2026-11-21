"""A cheaper programmer over cell-token states. Registered as PROGRAMMERS["token_pair_transformer"].

The lab's `token_query_transformer` lays the current state and the target state side by side as two tokens per cell, so its
transformer runs over twice as many tokens as the grid has cells (451 at 15x15) and takes most of a training step. This
part joins the two states cell by cell instead: the current and the target token of a cell are concatenated and projected
to one token, so the transformer runs over one token per cell plus the query (226). Attention costs a quarter and the rest
half. What a cell was and what it should become sit in the same token, so the comparison no longer has to be found by
attention. Everything else (2D RoPE layers, query token, output head, initialisation) is the lab part's.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from components.blocks.rope import RoPEEncoder
from components.blocks.transformer import init_linear_scaled
from core.registry import PROGRAMMERS


class TokenPairTransformer(nn.Module):
    def __init__(self, state_dim: int, d_model: int, action_dim: int, num_heads: int, d_ff: int, num_layers: int, grid_dim: int,
                 dropout: float = 0.0, layer_norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.join = nn.Linear(2 * state_dim, d_model)
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        self.transformer = RoPEEncoder(d_model, num_layers, num_heads, d_ff, dropout, grid_dim, tokens_per_cell=1)
        self.output_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.output = nn.Linear(d_model, action_dim, bias=False)
        self.apply(init_linear_scaled)
        nn.init.normal_(self.query, std=d_model**-0.5)

    def forward(self, current_state: Tensor, target_state: Tensor) -> Tensor:
        """current_state, target_state (B,L,state_dim) -> (B,1,action_dim)."""

        tokens = self.join(torch.cat((current_state, target_state), dim=-1))
        encoded = self.transformer(torch.cat((tokens, self.query.expand(tokens.shape[0], -1, -1)), dim=1))
        return self.output(self.output_norm(encoded[:, -1]))[:, None, :]


PROGRAMMERS["token_pair_transformer"] = TokenPairTransformer
