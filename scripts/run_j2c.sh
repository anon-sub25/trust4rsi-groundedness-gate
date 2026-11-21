#!/bin/bash
# J2c placebo control (docs/prereg-j2c.md). Cells C + D_u, bm25 k=10.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/bin/python3
mkdir -p logs
COMMON="--cells C D_u --fractions 0.1 0.25 0.5 0.75 1.0 --limit 40 --seed 20261107 --batch-size 1"
run () {
  local out="results/grid_${2}_J2c.jsonl"
  [ -f "$out" ] && [ "$(wc -l < "$out")" -eq 400 ] && { echo "### SKIP $2 (complete)"; return 0; }
  echo "### START $2 J2c $(date +%H:%M:%S)"
  PYTHONPATH=. $PY -m src.judges.local_runner --judge J2c --model "$1" $COMMON --out "$out" \
      > "logs/grid_${2}_J2c.log" 2>&1
  local rc=$? rows=0; [ -f "$out" ] && rows=$(wc -l < "$out")
  echo "### END   $2 J2c rc=$rc rows=$rows $(date +%H:%M:%S)"
  [ "$rc" -ne 0 ] && tail -6 "logs/grid_${2}_J2c.log"
  return 0
}
for pair in "Qwen/Qwen2.5-3B-Instruct qwen3b" \
            "ibm-granite/granite-3.3-8b-instruct granite8b" \
            "mistralai/Mistral-7B-Instruct-v0.3 mistral7b" \
            "Qwen/Qwen2.5-7B-Instruct qwen7b" \
            "microsoft/Phi-4-mini-instruct phi4mini"; do
  set -- $pair
  run "$1" "$2"
done
echo "### J2C COMPLETE $(date +%H:%M:%S)"
