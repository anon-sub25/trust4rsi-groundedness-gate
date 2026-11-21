#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### MISTRAL RAMP+ROLLBACK START $(date +%H:%M:%S)"
sysctl -n vm.page_free_count hw.memsize | awk 'NR==1{free=$1*16384} NR==2{total=$1} END{printf "System-wide memory free percentage: %d%%\n", free/total*100}'

../.venv/bin/python3 -m src.closed_loop.loop \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --arms R \
    --rollback \
    -f 0.25,0.5,0.75,1.0,1.0 \
    -k 10 \
    --rounds 5 \
    --out results/loop_mistral7b_ramp_rollback.jsonl

echo "### MISTRAL RAMP+ROLLBACK DONE $(date +%H:%M:%S)"
wc -l results/loop_mistral7b_ramp_rollback.jsonl | awk '{printf "mistral ramp+rollback rows: %8d (expect 4440)\n", $1}'

echo "### ANALYSIS"
../.venv/bin/python3 scripts/analyze_loop.py results/loop_mistral7b_ramp_rollback.jsonl
echo "### ALL DONE $(date +%H:%M:%S)"
