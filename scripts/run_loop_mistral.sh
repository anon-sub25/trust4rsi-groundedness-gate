#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### MISTRAL LOOP START $(date +%H:%M:%S)"
sysctl -n vm.page_free_count hw.memsize | awk 'NR==1{free=$1*16384} NR==2{total=$1} END{printf "System-wide memory free percentage: %d%%\n", free/total*100}'

../.venv/bin/python3 -m src.closed_loop.loop \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --arms A C \
    --rounds 5 \
    -f 0.25 \
    -k 10 \
    --out results/loop_mistral7b.jsonl

echo "### MISTRAL LOOP DONE $(date +%H:%M:%S)"
wc -l results/loop_mistral7b.jsonl | awk '{printf "mistral rows: %8d (expect 8880)\n", $1}'

echo "### ANALYSIS"
../.venv/bin/python3 scripts/analyze_loop.py results/loop_mistral7b.jsonl
echo "### ALL DONE $(date +%H:%M:%S)"
