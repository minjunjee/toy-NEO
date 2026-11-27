#!/usr/bin/env bash
# A per-task sleep over all tasks of a split, split over several processes on one GPU, merged at the end.
#   scripts/arc/per_task_sleep.sh <GPU uuid> <run dir> <out name> <shards> [arguments of sleep_per_task.py ...]
set -u
GPU=$1; RUN=$2; NAME=$3; SHARDS=$4; shift 4
cd "$(dirname "$0")/../.."
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 .venv/bin/python scripts/arc/sleep_per_task.py --run "$RUN" --shard $i/$SHARDS --out "$RUN/$NAME.json" "$@" > "$RUN/$NAME.shard$i.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
.venv/bin/python scripts/arc/sleep_per_task.py --merge $SHARDS --out "$RUN/$NAME.json" > "$RUN/$NAME.log" 2>&1
