#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python3
LOG=logs/framp_run.out

echo "### FRAMP RESUME $(date +%H:%M:%S)" | tee -a "$LOG"

PYTHONPATH=. $PY -m src.closed_loop.loop \
    --arms A D \
    --rounds 5 -f 0.1,0.25,0.5,0.75,1.0 -k 10 \
    --batch-size 1 --seed 20261107 \
    --resume \
    --out results/loop_qwen3b_framp.jsonl \
    >> "$LOG" 2>&1

echo "### FRAMP DONE $(date +%H:%M:%S)" | tee -a "$LOG"

FR=$(wc -l < results/loop_qwen3b_framp.jsonl)
echo "f-ramp rows: $FR (expect 8880)" | tee -a "$LOG"
[ "$FR" -eq 8880 ] || { echo "FATAL: f-ramp row count wrong" | tee -a "$LOG"; exit 1; }

echo "### ANALYSIS" | tee -a "$LOG"
$PY scripts/analyze_loop.py results/loop_qwen3b_framp.jsonl 2>&1 | tee -a "$LOG"
echo "### ALL DONE $(date +%H:%M:%S)" | tee -a "$LOG"
