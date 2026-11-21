#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### RAMP+ROLLBACK START $(date +%H:%M:%S)"
sysctl -n vm.page_free_count hw.memsize | awk 'NR==1{free=$1*16384} NR==2{total=$1} END{printf "System-wide memory free percentage: %d%%\n", free/total*100}'

../.venv/bin/python3 -m src.closed_loop.loop \
    --model Qwen/Qwen2.5-3B-Instruct \
    --arms R \
    --rollback \
    -f 0.25,0.5,0.75,1.0,1.0 \
    -k 10 \
    --rounds 5 \
    --out results/loop_qwen3b_ramp_rollback.jsonl

echo "### RAMP+ROLLBACK DONE $(date +%H:%M:%S)"
wc -l results/loop_qwen3b_ramp_rollback.jsonl | awk '{printf "ramp+rollback rows: %8d (expect 4440)\n", $1}'

echo "### ANALYSIS"
../.venv/bin/python3 scripts/analyze_loop.py results/loop_qwen3b_ramp_rollback.jsonl
echo "### ALL DONE $(date +%H:%M:%S)"
