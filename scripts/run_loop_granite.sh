#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### GRANITE LOOP START $(date +%H:%M:%S)"
sysctl -n vm.page_free_count hw.memsize | awk 'NR==1{free=$1*16384} NR==2{total=$1} END{printf "System-wide memory free percentage: %d%%\n", free/total*100}'

../.venv/bin/python3 -m src.closed_loop.loop \
    --model ibm-granite/granite-3.3-8b-instruct \
    --arms A C \
    --rounds 5 \
    -f 0.25 \
    -k 10 \
    --out results/loop_granite8b.jsonl

echo "### GRANITE LOOP DONE $(date +%H:%M:%S)"
wc -l results/loop_granite8b.jsonl | awk '{printf "granite rows: %8d (expect 8880)\n", $1}'

echo "### ANALYSIS"
../.venv/bin/python3 scripts/analyze_loop.py results/loop_granite8b.jsonl
echo "### ALL DONE $(date +%H:%M:%S)"
