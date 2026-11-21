#!/bin/bash
# Final runs for the AAAI submission.
# B2: de-confounded J2b arm (~15 min).
# f-ramp: arms A and D under growing memory (~30 min).
# Both share the model load.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python3
LOG=logs/final_runs.out

echo "### FINAL RUNS START $(date +%H:%M:%S)" | tee "$LOG"
memory_pressure 2>/dev/null | tail -1 | tee -a "$LOG"

# --- arm B2: de-confounded J2b ---
echo "### B2 START $(date +%H:%M:%S)" | tee -a "$LOG"
PYTHONPATH=. $PY -m src.closed_loop.loop \
    --arms B \
    --rounds 5 -f 0.25 -k 10 \
    --batch-size 1 --seed 20261107 \
    --abstention record \
    --out results/loop_qwen3b_b2.jsonl \
    >> "$LOG" 2>&1
echo "### B2 DONE $(date +%H:%M:%S)" | tee -a "$LOG"

# quick sanity
B2=$(wc -l < results/loop_qwen3b_b2.jsonl)
echo "B2 rows: $B2 (expect 4440)" | tee -a "$LOG"
[ "$B2" -eq 4440 ] || { echo "FATAL: B2 row count wrong" | tee -a "$LOG"; exit 1; }

# --- f-ramp arms A and D ---
echo "### FRAMP START $(date +%H:%M:%S)" | tee -a "$LOG"
PYTHONPATH=. $PY -m src.closed_loop.loop \
    --arms A D \
    --rounds 5 -f 0.1,0.25,0.5,0.75,1.0 -k 10 \
    --batch-size 1 --seed 20261107 \
    --out results/loop_qwen3b_framp.jsonl \
    >> "$LOG" 2>&1
echo "### FRAMP DONE $(date +%H:%M:%S)" | tee -a "$LOG"

FR=$(wc -l < results/loop_qwen3b_framp.jsonl)
echo "f-ramp rows: $FR (expect 8880)" | tee -a "$LOG"
[ "$FR" -eq 8880 ] || { echo "FATAL: f-ramp row count wrong" | tee -a "$LOG"; exit 1; }

echo "### ALL RUNS COMPLETE $(date +%H:%M:%S)" | tee -a "$LOG"

# Run analysis
echo "### ANALYSIS" | tee -a "$LOG"
/usr/bin/python3 scripts/analyze_loop.py results/loop_qwen3b_b2.jsonl 2>&1 | tee -a "$LOG"
echo "--- f-ramp ---" | tee -a "$LOG"
/usr/bin/python3 scripts/analyze_loop.py results/loop_qwen3b_framp.jsonl 2>&1 | tee -a "$LOG"
echo "### DONE $(date +%H:%M:%S)" | tee -a "$LOG"
