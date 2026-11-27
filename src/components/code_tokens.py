"""Parts for a code made of several vectors. Registered as PROGRAMMERS["token_pair_queries"] and EXECUTORS["token_context_transformer"].

The lab executor takes the code as one vector that scales and shifts every layer (AdaLN). That is one global setting for
the whole grid. Here the code is `num_queries` vectors and the executor reads them by attention: they are appended to the
cell tokens, so every cell can look up a different part of the code in every layer.

token_pair_queries          the pair programmer (`token_pair_transformer`) with several query tokens instead of one;
                            each query's output is one vector of the code. (B,L,d), (B,L,d) -> (B,num_queries,action_dim)
token_context_transformer   cell tokens followed by the code's vectors (each with a learned slot embedding), through the
                            lab's 2D RoPE layers (the code's vectors carry no position); the cell tokens that come out
                            are the next state. (B,L,state_dim), (B,num_code_tokens,cond_dim) -> (B,L,state_dim)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from components.blocks.rope import RoPEEncoder
from components.blocks.transformer import init_linear_scaled
from core.registry import EXECUTORS, PROGRAMMERS


class TokenPairQueries(nn.Module):
    def __init__(self, state_dim: int, d_model: int, action_dim: int, num_heads: int, d_ff: int, num_layers: int, grid_dim: int, num_queries: int,
                 dropout: float = 0.0, layer_norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.join = nn.Linear(2 * state_dim, d_model)
        self.query = nn.Parameter(torch.empty(1, num_queries, d_model))
        self.transformer = RoPEEncoder(d_model, num_layers, num_heads, d_ff, dropout, grid_dim, tokens_per_cell=1)
        self.output_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.output = nn.Linear(d_model, action_dim, bias=False)
        self.apply(init_linear_scaled)
        nn.init.normal_(self.query, std=d_model**-0.5)

    def forward(self, current_state: Tensor, target_state: Tensor) -> Tensor:
        tokens = self.join(torch.cat((current_state, target_state), dim=-1))
        encoded = self.transformer(torch.cat((tokens, self.query.expand(tokens.shape[0], -1, -1)), dim=1))
        return self.output(self.output_norm(encoded[:, tokens.shape[1]:]))


class TokenContextTransformer(nn.Module):
    def __init__(self, state_dim: int, cond_dim: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, grid_dim: int, num_code_tokens: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(state_dim, d_model)
        self.code_proj = nn.Linear(cond_dim, d_model)
        self.slot = nn.Parameter(torch.empty(1, num_code_tokens, d_model))
        self.transformer = RoPEEncoder(d_model, num_layers, num_heads, d_ff, dropout, grid_dim, tokens_per_cell=1)
        self.out_proj = nn.Linear(d_model, state_dim)
        self.apply(init_linear_scaled)
        nn.init.normal_(self.slot, std=d_model**-0.5)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        cells = state.shape[1]
        tokens = torch.cat((self.in_proj(state), self.code_proj(action) + self.slot), dim=1)
        return self.out_proj(self.transformer(tokens)[:, :cells])


PROGRAMMERS["token_pair_queries"] = TokenPairQueries
EXECUTORS["token_context_transformer"] = TokenContextTransformer
