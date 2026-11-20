#!/usr/bin/env bash
# Generate BabyTheorist files for one suite directory with the benchmark's own generator and its seed convention
# (train seed 42, eval seed 43, pool seed 42, eval files exclude everything already in the directory).
#
#   scripts/babytheorist/generate_suite.sh <benchmark mode> <output directory name> "<levels>" <n train> <n eval> [extra generator args]
#
# BT_HOME is a checkout of github.com/ahn-ml/babytheorist. Files go to data/babytheorist/<output directory name>/.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BT_HOME="${BT_HOME:-/home/minjun_dev/neo-vocabulary-restructuring/runs/external/babytheorist}"
PY="${PYTHON:-$REPO/.venv/bin/python}"
MODE=$1; NAME=$2; LEVELS=$3; N_TRAIN=$4; N_EVAL=$5; shift 5
OUT="$REPO/data/babytheorist/$NAME"
mkdir -p "$OUT"
cd "$BT_HOME"
for level in $LEVELS; do
  for split in train eval; do
    file="$OUT/lv${level}_${split}.h5"
    [[ -s "$file" ]] && { echo "exists: $file"; continue; }
    if [[ $split == train ]]; then n=$N_TRAIN; seed=42; else n=$N_EVAL; seed=43; fi
    echo "$(TZ=Asia/Seoul date '+%F %T KST') start $NAME level $level $split n=$n"
    "$PY" generate.py --level "$level" --n "$n" --mode "$split" --world-size 15 --shape-profile standard --benchmark-mode "$MODE" \
      --max-support 5 --backend filter --workers "${WORKERS:-16}" --seed "$seed" --pool-seed 42 \
      --dominant-pool-dir "$OUT/dominant_pools" --exclude "$OUT" --out "$file" "$@" > "$OUT/lv${level}_${split}.log" 2>&1
    echo "$(TZ=Asia/Seoul date '+%F %T KST') done  $NAME level $level $split"
  done
done
