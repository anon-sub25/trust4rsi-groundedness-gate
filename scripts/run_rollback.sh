#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python3
LOG=logs/rollback_run.out

echo "### ROLLBACK START $(date +%H:%M:%S)" | tee "$LOG"
memory_pressure 2>/dev/null | tail -1 | tee -a "$LOG" || true

PYTHONPATH=. $PY -m src.closed_loop.loop \
    --arms R \
    --rounds 5 -f 0.25 -k 10 \
    --batch-size 1 --seed 20261107 \
    --rollback \
    --out results/loop_qwen3b_rollback.jsonl \
    >> "$LOG" 2>&1

echo "### ROLLBACK DONE $(date +%H:%M:%S)" | tee -a "$LOG"

RB=$(wc -l < results/loop_qwen3b_rollback.jsonl)
echo "rollback rows: $RB (expect 4440)" | tee -a "$LOG"
[ "$RB" -eq 4440 ] || { echo "FATAL: rollback row count wrong" | tee -a "$LOG"; exit 1; }

echo "### ANALYSIS" | tee -a "$LOG"
$PY scripts/analyze_loop.py results/loop_qwen3b_rollback.jsonl 2>&1 | tee -a "$LOG"
echo "### ALL DONE $(date +%H:%M:%S)" | tee -a "$LOG"
