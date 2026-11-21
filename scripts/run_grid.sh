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

# J2b first in each pair: it is the configuration that lifts cell C off the
# floor on Granite, so it is the one carrying a measurable d-prime.
for pair in "ibm-granite/granite-3.3-8b-instruct granite8b" \
            "Qwen/Qwen2.5-7B-Instruct qwen7b" \
            "Qwen/Qwen2.5-3B-Instruct qwen3b"; do
  set -- $pair
  run "$1" J2b "$2"
  run "$1" J2a "$2"
done
echo "### PHASE 1 COMPLETE $(date +%H:%M:%S)"
