#!/bin/bash
# Closed-loop ratchet (docs/prereg-ratchet.md).
# Waits for any running judge job so only one model is resident at a time.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "run_j2c.sh|run_composition.sh|run_grid" > /dev/null; do
  echo "waiting for the running judge job... $(date +%H:%M:%S)"
  sleep 60
done
echo "### LOOP START $(date +%H:%M:%S)"
PYTHONPATH=. ../.venv/bin/python3 -m src.closed_loop.loop \
    --model Qwen/Qwen2.5-3B-Instruct --arms A B C D --rounds 5 \
    -f 0.25 -k 10 --batch-size 1 --seed 20261107 \
    --out results/loop_qwen3b.jsonl
echo "### LOOP END rc=$? $(date +%H:%M:%S)"
[ -f results/loop_qwen3b.jsonl ] && echo "rows: $(wc -l < results/loop_qwen3b.jsonl)"
