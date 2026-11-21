#!/usr/bin/env python3
"""What the grid looks like between the steps of a program, and what the next step does with it.

For the eval tasks of one primitive that the model explains with exactly the given code sequence, this decodes the grid
after the first code and names it (the input, the target, the object flipped upside-down, left-right or turned 180 degrees
in place). For a two-code program it then runs the second code twice: from the encoding of the decoded cell probabilities
(what the lab's reencode_soft hands over) and from the encoding of the decoded grid itself. If the two differ, the second
step is using something that is not visible on the grid.

    python scripts/babytheorist/intermediate_states.py <run name> <primitive> <codes, comma separated>
    python scripts/babytheorist/intermediate_states.py neo_transformer_36k_codes8_sleep14400_single_a_lv1_s42 flip_v 3,3
"""

import collections
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from training.evaluate import load_run  # noqa: E402

def in_place(grid, fn):
    out = grid.clone()
    for b in range(grid.shape[0]):
        nz = torch.nonzero(grid[b])
        r0, c0, r1, c1 = nz[:, 0].min(), nz[:, 1].min(), nz[:, 0].max() + 1, nz[:, 1].max() + 1
        out[b, r0:r1, c0:c1] = fn(grid[b, r0:r1, c0:c1])
    return out

run, primitive, program = sys.argv[1], sys.argv[2], [int(c) for c in sys.argv[3].split(',')]
device = torch.device('cuda')
model, data, config, step = load_run(REPO / 'runs' / run, None, device)
infer = {'max_steps': 5, 'rollout': 'carry'}
names = collections.Counter(); conf_mid, conf_end, n = 0.0, 0.0, 0
hard_second = collections.Counter(); soft_second = collections.Counter()
with torch.no_grad():
    for batch in data.eval_loader('lv1', 500):
        keep = [i for i in torch.nonzero(batch['depth'] == 1).flatten().tolist() if json.loads(batch['transitions'][i])[0]['primitive'] == primitive]
        if not keep: continue
        x, y = batch['support_x'][keep].to(device), batch['support_y'][keep].to(device)
        out = model.self_explain(x, y, **infer)
        codes = torch.stack([s.view(len(keep), -1)[:, 0] for s in out['indices']], 1)
        ok = out['pred'].eq(y).flatten(1).all(1) & (out['n_steps'] == len(program))
        for k, c in enumerate(program): ok &= codes[:, k] == c
        if not ok.any(): continue
        x, y = x[ok], y[ok]
        vec = model.code.codebook(torch.tensor(program, device=device))
        s1 = model.executor(model.encoder(x), vec[0][None].expand(len(x), -1, -1) if vec[0].dim() == 2 else vec[0].expand(len(x), *vec[0].shape))
        logits1 = model.decoder(s1); p1 = torch.softmax(logits1, -1); g1 = logits1.argmax(-1)
        cands = {'input': x, 'target': y, 'flip upside-down': in_place(x, lambda o: o.flip(0)), 'flip left-right': in_place(x, lambda o: o.flip(1)), 'rot180': in_place(x, lambda o: o.flip(0).flip(1))}
        for b in range(len(x)):
            hit = [k for k, g in cands.items() if torch.equal(g[b], g1[b])]
            names[', '.join(hit) if hit else 'none of these'] += 1
        obj = (x != 0) | (y != 0) | (g1 != 0)
        conf_mid += float((p1.max(-1).values * obj).sum(dim=(1, 2)).div(obj.sum(dim=(1, 2))).sum())
        if len(program) > 1:
            v2 = vec[1][None].expand(len(x), -1, -1) if vec[1].dim() == 2 else vec[1].expand(len(x), *vec[1].shape)
            soft = model.decoder(model.executor(model.encoder(p1), v2)); hard = model.decoder(model.executor(model.encoder(g1), v2))
            conf_end += float((torch.softmax(soft, -1).max(-1).values * obj).sum(dim=(1, 2)).div(obj.sum(dim=(1, 2))).sum())
            for b in range(len(x)):
                for counter, g in ((soft_second, soft.argmax(-1)), (hard_second, hard.argmax(-1))):
                    hit = [k for k, c in cands.items() if torch.equal(c[b], g[b])] + (['same as after step 1'] if torch.equal(g[b], g1[b]) else [])
                    counter[', '.join(hit) if hit else 'none of these'] += 1
        n += len(x)
print(run, primitive, program, 'tasks', n)
print(' grid after step 1 equals:', dict(names))
print(' mean confidence on object cells after step 1: %.3f' % (conf_mid / max(n, 1)), ('after step 2 (soft): %.3f' % (conf_end / max(n, 1))) if len(program) > 1 else '')
if len(program) > 1:
    print(' step 2 from the soft state gives:', dict(soft_second))
    print(' step 2 from the hard grid gives :', dict(hard_second))
