"""Fixed training loop: DDP, autocast, batches, method.train_step, logging, checkpoints, resume, evals."""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from core.config import TrainConfig, train_config
from core.registry import DATASETS, EVALS, METHODS, build, import_all


@dataclass(frozen=True)
class Runtime:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


def runtime() -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return Runtime(rank, local_rank, world_size, device)


def git_state(path: Path) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=path, text=True, stderr=subprocess.DEVNULL)
        return {"commit": commit, "dirty": bool(status.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"])


def run_evals(model, data, blocks: list[dict], step: int, device: torch.device, eval_dir: Path | None = None) -> dict[str, float]:
    """Run every block; keys are ``<type>/<split>/<metric>``. Blocks are already free of 'every'."""

    model.eval()
    state = rng_state()
    metrics: dict[str, float] = {}
    try:
        for block in blocks:
            block = dict(block)
            eval_type = block.pop("type")
            name = block.pop("name", eval_type)
            block.pop("every", None)
            result = EVALS[eval_type](model, data, block, step, device)
            prefix = f"{name}/{block['split']}"
            metrics.update({f"{prefix}/{k}": v for k, v in result.items()})
            if eval_dir is not None:
                eval_dir.mkdir(parents=True, exist_ok=True)
                (eval_dir / f"{name}_{block['split']}_step{step:06d}.json").write_text(
                    json.dumps({"step": step, "config": block, **result}, indent=2), encoding="utf-8"
                )
    finally:
        set_rng_state(state)
    return metrics


def save_checkpoint(path: Path, model, method, step: int, config: dict, git: dict, raw=None) -> None:
    """With an EMA, "model" holds the averaged weights (used by eval and init_from) and "raw_model" the trained ones."""
    module = model.module if isinstance(model, DistributedDataParallel) else model
    payload = {"step": step, "model": module.state_dict(), "method": method.state_dict(), "config": config, "git": git, "rng": rng_state()}
    if raw is not None:
        payload["raw_model"] = raw.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config: dict, output_dir: Path, resume: Path | None = None) -> Path | None:
    import_all()
    rt = runtime()
    tc: TrainConfig = train_config(config["train"])
    if tc.batch_size % rt.world_size:
        raise ValueError("batch_size must be divisible by the world size")
    seed_everything(tc.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    data = build(DATASETS, config["data"])
    method = build(METHODS, config["method"])
    model = method.build_model(config["model"], tc.init_from).to(rt.device)
    method.attach(model, rt.device, tc.precision)
    start_step = 0
    checkpoint = None
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("raw_model", checkpoint["model"]))
        method.load_state_dict(checkpoint["method"])
        start_step = int(checkpoint["step"])
    if rt.world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rt.local_rank], find_unused_parameters=True)
    module = model.module if isinstance(model, DistributedDataParallel) else model
    ema = None
    if tc.ema:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(module, multi_avg_fn=get_ema_multi_avg_fn(tc.ema), use_buffers=True)
        if checkpoint is not None and "raw_model" in checkpoint:
            ema.module.load_state_dict(checkpoint["model"])
    averaged = ema.module if ema is not None else module
    if tc.compile == "parts" and rt.device.type == "cuda":
        # after the averaged copy is made, so the copy used for evaluation and checkpoints stays a plain module; the forward
        # of the trained parts is replaced by its compiled version, parameters and state_dict keys are untouched
        for name in ("programmer", "executor"):
            part = getattr(module, name, None)
            if part is not None:
                part.forward = torch.compile(part.forward)

    loader = data.train_loader(tc.batch_size, tc.steps, start_step, tc.seed, rt.rank, rt.world_size, tc.num_workers, rt.device.type == "cuda")
    git = git_state(Path(__file__).resolve().parents[2])
    run = None
    log_file = None
    if rt.primary:
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved = {
            **config, "host": socket.gethostname(), "git": git, "torch": torch.__version__,
            "cuda": torch.version.cuda, "world_size": rt.world_size, "output_dir": str(output_dir),
            "init_report": getattr(method, "init_report", None),
        }
        (output_dir / "resolved.json").write_text(json.dumps(resolved, indent=2, default=str), encoding="utf-8")
        log_file = (output_dir / "train.jsonl").open("a", encoding="utf-8")
        if tc.wandb_mode != "disabled":
            import wandb

            run = wandb.init(project=tc.wandb_project, group=config["model"]["type"], name=output_dir.name, dir=str(output_dir), mode=tc.wandb_mode, config=resolved)
    if checkpoint is not None:
        set_rng_state(checkpoint["rng"])

    method.set_train_mode(model)
    started = time.monotonic()
    last_log = (started, start_step)
    final_path = None
    for step, batch in enumerate(loader, start=start_step):
        batch = to_device(batch, rt.device)
        metrics = method.train_step(model, batch, step)
        if ema is not None:
            ema.update_parameters(module)
        completed = step + 1
        if completed % tc.log_every == 0 or completed == tc.steps:
            if rt.world_size > 1:
                names = sorted(metrics)
                values = torch.tensor([metrics[n] for n in names], device=rt.device)
                dist.all_reduce(values, op=dist.ReduceOp.AVG)
                metrics = dict(zip(names, values.tolist()))
            now = time.monotonic()
            metrics.update(step=completed, samples_per_second=(completed - last_log[1]) * tc.batch_size / (now - last_log[0]), elapsed=now - started)
            last_log = (now, completed)
            if rt.primary:
                line = json.dumps(metrics)
                print(line, flush=True)
                log_file.write(line + "\n")
                log_file.flush()
                if run is not None:
                    run.log(metrics, step=completed)
        if completed % tc.ckpt_every == 0 or completed == tc.steps:
            if rt.primary:
                final_path = output_dir / "ckpt" / f"step-{completed:06d}.pt"
                save_checkpoint(final_path, averaged, method, completed, config, git, raw=module if ema is not None else None)
            if rt.world_size > 1:
                dist.barrier()
        due = [b for b in config["evals"] if completed % b["every"] == 0 or completed == tc.steps]
        if due:
            if rt.primary:
                eval_metrics = run_evals(averaged, data, due, completed, rt.device, output_dir / "eval")
                line = json.dumps({"step": completed, **eval_metrics})
                print(line, flush=True)
                log_file.write(line + "\n")
                log_file.flush()
                if run is not None:
                    run.log(eval_metrics, step=completed)
            method.set_train_mode(model)
            if rt.world_size > 1:
                dist.barrier()

    if rt.primary:
        log_file.close()
        if run is not None:
            run.finish()
    if rt.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return final_path
