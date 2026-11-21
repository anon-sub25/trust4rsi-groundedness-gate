#!/bin/bash
# Full grid: 5 cells x 5 fractions x n=40 = 1000 scorings per run.
# Supersedes the Nov 16 C/D_u-only runs, which stay on disk as a subset check.
# Strictly sequential -- one model resident at a time on a 48 GB machine.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/bin/python3
COMMON="--cells C C_v D_u E E_v --fractions 0.1 0.25 0.5 0.75 1.0 --limit 40 --seed 20261107 --batch-size 1"

run () {   # $1 = model, $2 = judge, $3 = tag
  local out="results/grid_${3}_${2}.jsonl"
  [ -f "$out" ] && [ "$(wc -l < "$out")" -eq 1000 ] && { echo "### SKIP $3 $2 (complete)"; return 0; }
  echo "### START $3 $2 $(date +%H:%M:%S)"
  PYTHONPATH=. $PY -m src.judges.local_runner \
      --judge "$2" --model "$1" $COMMON --out "$out" \
      > "logs/grid_${3}_${2}.log" 2>&1
  local rc=$? rows=0
  [ -f "$out" ] && rows=$(wc -l < "$out")
  echo "### END   $3 $2 rc=$rc rows=$rows $(date +%H:%M:%S)"
  [ "$rc" -ne 0 ] && tail -4 "logs/grid_${3}_${2}.log"
  return 0
}

# Phase 2: families that fit in bf16 here. Measured pre-load headroom is
# 26.9 GB, so Qwen-14B (29.6 GB) does not fit and Mistral-Nemo-12B (24.5 GB)
# leaves ~2.4 GB for activations and KV cache -- it would load and then swap.
# Excluded by measurement, not estimate. 4-bit is no escape: bitsandbytes is
# CUDA-only, so --load-4bit is inert on MPS.
#
# Run as its own script, after run_grid.sh. Appending these lines to a script
# bash was already executing would have worked only by accident of byte
# offsets.
for pair in "mistralai/Mistral-7B-Instruct-v0.3 mistral7b" \
            "microsoft/Phi-4-mini-instruct phi4mini"; do
  set -- $pair
  while pgrep -f prefetch_weights.py >/dev/null; do sleep 30; done
  run "$1" J2b "$2"
  run "$1" J2a "$2"
done
echo "### PHASE 2 COMPLETE $(date +%H:%M:%S)"
