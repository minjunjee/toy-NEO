# GeneralizedL2T

A config-driven codebase for NEO (learning-to-theorize) on GridWorld, with a transformer NEO baseline.
A run is one yaml; every part of it (data, model parts, training method, evaluation) is picked by name.

## Quick start

```
python scripts/generate.py --spec configs/data/paper-alpha-0.33.yaml            # data (writes to the spec's output dir)
python scripts/train.py --config configs/runs/ae_tok_gw_s42.yaml                 # codec pretraining
torchrun --nproc-per-node 2 scripts/train.py --config configs/runs/neo_transformer_gw033_s42.yaml
python scripts/finish_run.py --run runs/neo_transformer_gw033_s42               # best checkpoint + its evals
python scripts/evaluate.py --run runs/neo_transformer_gw033_s42 [--ckpt ...]    # re-run evals
```

Env: python 3.10, torch 2.8. Outputs go to `runs/<config name>/` (config, `train.jsonl`, `ckpt/`, `eval/`).

## Models

NEO explains a pair (x, y) as a short sequence of discrete codes, picks the length by an MDL rule, and transfers the
codes from a support pair to a query input. The transformer baseline changes the parts and the state passed between
steps; the length rule stays the official one.

| | NEO (official, `neo_gw033_k4_cb6_s*`) | transformer NEO (`neo_transformer_gw033_s*`) |
|---|---|---|
| codec | CNN VAE, one 32-d latent per grid | one 128-d token per cell (`token_embed` / `linear_head`) |
| programmer | MLP on (s_k, s_y) | 6-layer transformer with 2D RoPE over cell tokens |
| code | VQ, 6 codes, sampled while training | VQ, 16 codes, nearest code, commitment gradient on |
| executor | FiLM MLP, residual | 4-layer AdaLN-Zero transformer with 2D RoPE |
| between steps | latent carried as is | soft re-encode: E(softmax D(s_k)) |
| length L* | argmin 0.95^k x CE, shortest exact wins (at most 4 steps) | same (at most 5 steps) |
| loss | CE at L* + 0.1 grounding + VQ at L* | CE at L* + 0.1 grounding + VQ averaged over steps 1..L* |
| training | lr 5e-4, batch 128 episodes, 39,100 steps | lr 3e-4, batch 256 episodes, 78,200 steps, weight EMA 0.999 |

Both freeze the pretrained codec. The transformer NEO is evaluated with `rollout: carry` (support length from the
training rule, the query replays it with the same state update); the query target is only scored.

## Results on alpha-0.33

Transfer exact match, ID / Comp / Len (5,000 / 5,000 / 10,000 episodes), mean over three seeds or runs.

| model | ID | Comp | Len |
|---|---|---|---|
| NEO, paper Table 1 | 0.911 | 0.933 | 0.845 |
| NEO, learning-to-theorize release 1c190c8 | 0.969 | 0.955 | 0.888 |
| NEO in this repo (paper evaluator) | 0.902 | 0.894 | 0.811 |
| transformer NEO (best ID self-explanation; runs stopped at step ~23,500 of 78,200) | 0.999 | 0.968 | 0.954 |

## Code layout

```
src/components/  encoders, decoders, programmers, codes, executors   (model parts)
src/models/      neo, vae                                             (rollouts)
src/methods/     neo, vae_pretrain                                    (loss, optimizer, schedule)
src/evals/       self_explain_exact, transfer_exact
src/data/        gridworld_pairs loader, GridWorld generator and profiles
src/training/    trainer (DDP, bf16, EMA, checkpoints), best-checkpoint selection
```

Each module registers its classes by name (`EXECUTORS["adaln_transformer"] = AdaLNTransformer`), and a config block
`{type: <name>, ...}` builds that class with the other keys as arguments; unknown keys are errors. To add something,
write the class, register it, and use its name in a config (a new module file also goes into `core.registry.import_all`).

| registry | names |
|---|---|
| encoders / decoders | `cnn_vae`, `token_embed`, `vit_latent` / `cnn_vae`, `linear_head`, `vit_latent` |
| programmers | `film_mlp`, `token_query_transformer` |
| codes | `vq`, `gaussian` |
| executors | `film_mlp`, `adaln_transformer` |
| models / methods | `neo`, `vae` / `neo`, `vae_pretrain` |
| evals | `self_explain_exact`, `transfer_exact` |
| datasets | `gridworld_pairs` (`sampling: episode` or `pair`) |

Useful `neo` options: `length_cost` (+ `length_cost_ramp`) for a per-code length cost, `code_loss_scope`
(`selected` = VQ at L* only, `prefix` = steps 1..L*), `code.stochastic`, `code.commitment_grad`, eval `rollout`
(`official` / `carry` / `hard`) and `transfer_exact` `group_by_program`.

`tests/test_configs.py` builds every config, takes one train step and runs its evals on CPU; `tests/test_parity_neo.py`
checks NEO against the official code.

Sources: learning-to-theorize (2d7d477) for NEO, World-Theory-Model (840b0cb) for the cell-token codec and the pair
sampler, BabyTheorist (e51a7d1) for the AdaLN-Zero and RoPE parts.
