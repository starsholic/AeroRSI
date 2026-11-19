#!/usr/bin/env bash
# The paper's main experiment: three rounds of AeroRSI from the supervised policy, each evaluated on the official
# test splits.
#   GPUS="0 1 2 3" OUT=outputs/aerorsi bash scripts/run_loop.sh
#   INIT=checkpoints/AeroDPO-2B-384-LoRA OUT=outputs/aerorsi_dpo bash scripts/run_loop.sh   # DPO init.
# Navigation stages (rollout, verify, evaluate) run simulator workers on all of GPUS; relabel and train use TRAIN_GPU.
# A stage refuses to write into an existing directory, so a finished stage is skipped on re-runs.
set -euo pipefail
GPUS=${GPUS:-"0 1 2 3"}
TRAIN_GPU=${TRAIN_GPU:-${GPUS%% *}}
OUT=${OUT:-outputs/aerorsi}
ROUNDS=${ROUNDS:-3}

stage() {  # stage <output dir> <aerorsi arguments...>
  local dir=$1; shift
  if [[ -f "$dir/metrics.json" ]]; then echo "== skip $dir (done)"; return; fi
  if [[ -e "$dir" ]]; then echo "$dir exists but is unfinished; inspect it, then remove it to rerun" >&2; exit 1; fi
  echo "== $dir"; aerorsi "$@" --output "$dir"
}

if [[ -n "${INIT:-}" ]]; then  # e.g. INIT=checkpoints/AeroDPO-2B-384-LoRA: start from an existing adapter (DPO init.)
  ADAPTER=$INIT
else                          # the supervised policy itself (SFT init.)
  stage "$OUT/round0" init-adapter
  ADAPTER=$OUT/round0/adapter
fi
REPLAY=()
for k in $(seq 1 "$ROUNDS"); do
  R=$OUT/round$k
  # The current policy flies every training route; its failures give this round's anchors.
  stage "$R/rollout"  rollout  --adapter "$ADAPTER" --gpus $GPUS
  # Normal states for retention come once from the round-0 policy's own successful flights.
  if [[ $k == 1 ]]; then stage "$OUT/normal" normal-inputs --rollouts "$R/rollout"; fi
  stage "$R/anchors"  anchor   --rollouts "$R/rollout" --normal-inputs "$OUT/normal/normal_inputs.jsonl"
  # It verifies candidate repairs by flying on from each of them.
  stage "$R/verify"   verify   --plan "$R/anchors" --adapter "$ADAPTER" --gpus $GPUS
  stage "$R/repairs"  pool     --collection "$R/verify"
  # Its own outputs on normal states are the retention targets.
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU stage "$R/relabel" relabel --adapter "$ADAPTER" --inputs "$R/anchors/normal_train.jsonl"
  stage "$R/rows"     assemble --repairs "$R/repairs" --plan "$R/anchors" --predictions "$R/relabel" ${REPLAY[@]+"${REPLAY[@]}"}
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU stage "$R/train" train --rows "$R/rows" --adapter "$ADAPTER"
  ADAPTER=$R/train/final
  REPLAY=(--replay "$R/rows")
  stage "$R/test"     evaluate --adapter "$ADAPTER" --gpus $GPUS
  aerorsi metrics --runs "$R/test"
done
