"""Parts for a code that has a vector and a map. Registered as PROGRAMMERS["token_pair_map"] and EXECUTORS["adaln_map_transformer"].

The lab code is one vector that scales and shifts every layer of the executor: one global setting for the whole grid. A
rule that needs detail, such as a pattern to stamp or a table of which colour becomes which, has to be squeezed into it.
Here the code keeps that vector and adds a map: one small vector for every cell of the canvas, the same shape as a state.
The executor reads the map by attention, so each cell can look up the part of the code it needs.

token_pair_map         the pair programmer (`token_pair_transformer`) with a second head: the query token gives the
                       vector as before, and every cell token gives one vector of the map.
                       (B,L,d), (B,L,d) -> (B,1,action_dim + L*map_dim), vector first, then the map cell by cell
adaln_map_transformer  the lab's AdaLN-Zero executor with one more step in every block, between self-attention and the
                       feed-forward: the cell tokens attend to the map (2D RoPE on both sides, since map and state share
                       the canvas). That step's output projection starts at zero, so at initialisation the part is the
                       lab's `adaln_transformer`, and a checkpoint of that part loads into this one unchanged.
                       (B,L,state_dim), (B,1,cond_dim + L*map_dim) -> (B,L,state_dim)

Everywhere else the code is one flat vector, so averaging over pairs, fitting at test time and the sleeps work unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from components.blocks.adaln import AdaLNZeroBlock
from components.blocks.rope import RoPEEncoder
from components.blocks.transformer import init_linear_scaled
from components.positional import apply_rope, rope_2d
from core.registry import EXECUTORS, PROGRAMMERS


class TokenPairMap(nn.Module):
    def __init__(self, state_dim: int, d_model: int, action_dim: int, num_heads: int, d_ff: int, num_layers: int, grid_dim: int, map_dim: int,
                 dropout: float = 0.0, layer_norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.join = nn.Linear(2 * state_dim, d_model)
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        self.transformer = RoPEEncoder(d_model, num_layers, num_heads, d_ff, dropout, grid_dim, tokens_per_cell=1)
        self.output_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.output = nn.Linear(d_model, action_dim, bias=False)
        self.map_output = nn.Linear(d_model, map_dim, bias=False)
        self.apply(init_linear_scaled)
        nn.init.normal_(self.query, std=d_model**-0.5)

    def forward(self, current_state: Tensor, target_state: Tensor) -> Tensor:
        tokens = self.join(torch.cat((current_state, target_state), dim=-1))
        encoded = self.output_norm(self.transformer(torch.cat((tokens, self.query.expand(tokens.shape[0], -1, -1)), dim=1)))
        return torch.cat((self.output(encoded[:, -1]), self.map_output(encoded[:, :-1]).flatten(1)), dim=1)[:, None, :]


class AdaLNMapBlock(AdaLNZeroBlock):
    """The lab's AdaLN-Zero block with attention from the cell tokens to the code's map between self-attention and feed-forward."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, grid_dim: int) -> None:
        super().__init__(d_model, num_heads, d_ff, dropout, grid_dim)
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.map_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.map_query = nn.Linear(d_model, d_model)
        self.map_key_value = nn.Linear(d_model, 2 * d_model)
        self.map_out = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.map_out.weight)
        nn.init.zeros_(self.map_out.bias)
        cos, sin = rope_2d(grid_dim, self.head_dim)
        self.register_buffer("map_cos", cos, persistent=False)
        self.register_buffer("map_sin", sin, persistent=False)

    def forward(self, x: Tensor, c: Tensor, code_map: Tensor) -> Tensor:
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN_modulation(c).unsqueeze(1).chunk(6, dim=-1)
        x = x + alpha1 * self.attn((1 + gamma1) * self.norm1(x) + beta1)
        batch, cells, d_model = x.shape
        heads = lambda t: t.view(batch, cells, self.num_heads, self.head_dim).transpose(1, 2)  # noqa: E731
        key, value = self.map_key_value(code_map).chunk(2, dim=-1)
        query = apply_rope(heads(self.map_query(self.map_norm(x))), self.map_cos, self.map_sin)
        read = F.scaled_dot_product_attention(query, apply_rope(heads(key), self.map_cos, self.map_sin), heads(value))
        x = x + self.map_out(read.transpose(1, 2).reshape(batch, cells, d_model))
        return x + alpha2 * self.ffn((1 + gamma2) * self.norm2(x) + beta2)


class AdaLNMapStack(nn.Module):
    """Same attribute names as the lab's AdaLNZeroStack, so its checkpoints load here; the map's parts are extra."""

    def __init__(self, token_dim: int, cond_dim: int, map_dim: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float, grid_dim: int) -> None:
        super().__init__()
        self.cond_dim, self.map_dim = cond_dim, map_dim
        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.in_proj = nn.Linear(token_dim, d_model) if d_model != token_dim else nn.Identity()
        self.out_proj = nn.Linear(d_model, token_dim) if d_model != token_dim else nn.Identity()
        self.map_proj = nn.Linear(map_dim, d_model)
        self.map_proj_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(AdaLNMapBlock(d_model, num_heads, d_ff, dropout, grid_dim) for _ in range(num_layers))

    def forward(self, tokens: Tensor, code: Tensor) -> Tensor:
        code = code.reshape(code.shape[0], -1)
        c = self.cond_proj(code[:, : self.cond_dim])
        code_map = self.map_proj_norm(self.map_proj(code[:, self.cond_dim:].reshape(code.shape[0], tokens.shape[1], self.map_dim)))
        x = self.in_proj(tokens)
        for block in self.blocks:
            x = block(x, c, code_map)
        return self.out_proj(x)


class AdaLNMapTransformer(nn.Module):
    def __init__(self, state_dim: int, cond_dim: int, map_dim: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, grid_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.stack = AdaLNMapStack(state_dim, cond_dim, map_dim, d_model, num_layers, num_heads, d_ff, dropout, grid_dim)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.stack(state, action)


PROGRAMMERS["token_pair_map"] = TokenPairMap
EXECUTORS["adaln_map_transformer"] = AdaLNMapTransformer
