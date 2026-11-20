#!/bin/bash
# Usage: sbatch --job-name <run> --gres gpu:<n> [--dependency afterok:<id>] scripts/slurm/run.sbatch configs/runs/<run>.yaml
# turing hosts are not on the mlml sheet (set/clear refused); slurm allocation is the registration.
#SBATCH --partition gpu_jobs
#SBATCH --nodelist turing-1
#SBATCH --cpus-per-task 20
#SBATCH --output runs/%x/slurm-%j.out
set -euo pipefail
CONFIG=$1
REPO=/data/tgkang/projects/activated/generalizedL2T/codebases/GeneralizedL2T
cd "$REPO"
source /data/tgkang/miniconda3/etc/profile.d/conda.sh
conda activate l2t
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NGPU=${#GPUS[@]}
echo "host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-} config=$CONFIG start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nproc-per-node "$NGPU" scripts/train.py --config "$CONFIG"
else
  python scripts/train.py --config "$CONFIG"
fi
echo "end=$(date -Is)"
