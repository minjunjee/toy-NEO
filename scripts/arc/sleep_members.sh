#!/usr/bin/env bash
# Per-task sleeps of several trained models, one after another on one GPU, each split over shards.
#   scripts/arc/sleep_members.sh <GPU uuid> <shards> <list file>
# Every line of the list: <run dir> <checkpoint> <result name> [arguments of sleep_per_task.py ...]
# Results go to <run dir>/sleep/<result name>.json; a member whose result exists is skipped.
set -u
GPU=$1; SHARDS=$2; LIST=$3
cd "$(dirname "$0")/../.."
while read -r RUN CKPT NAME REST; do
  [ -z "${RUN:-}" ] && continue
  case "$RUN" in \#*) continue;; esac
  mkdir -p "$RUN/sleep"
  if [ -f "$RUN/sleep/$NAME.json" ]; then echo "$(TZ=Asia/Seoul date '+%F %T') skip $RUN $NAME"; continue; fi
  until [ -f "$CKPT" ]; do sleep 60; done
  echo "$(TZ=Asia/Seoul date '+%F %T') start $RUN $NAME"
  # shellcheck disable=SC2086
  scripts/arc/per_task_sleep.sh "$GPU" "$RUN" "sleep/$NAME" "$SHARDS" --checkpoint "$CKPT" --split evaluation --compile --max-pairs 48 $REST
  echo "$(TZ=Asia/Seoul date '+%F %T') done $RUN $NAME: $(grep -h '"vote2"' "$RUN/sleep/$NAME.log" | head -n 1)"
done < "$LIST"
