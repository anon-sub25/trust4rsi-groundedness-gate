# Groundedness Gate Audit

Code and results for "Can a Groundedness Judge Safely Gate Agent Memory?" (Trust4RSI-Agent @ AAAI 2027).

## Structure

```
src/
  judges/          Judge prompt loading, local model runner, contrast scoring
  closed_loop/     Multi-round loop simulator (arms A-D, ramp, rollback)
  testbed/         Raw LoCoMo store loader
  analysis/sdt.py  Signal detection theory (d', criterion, rates)
  build_cells.py   Stimulus construction from LoCoMo
  data_prep.py     LoCoMo preprocessing
  memory_axis.py   BM25 retriever for memory-level manipulation
  render.py        Unit rendering for judge prompts

scripts/
  run_grid*.sh           Static grid (5 models x 3 judges x 5 memory levels)
  run_loop*.sh           Closed-loop arms A-D (Qwen, Granite, Mistral)
  run_ramp*.sh           Ramp and ramp+rollback interventions
  run_rollback.sh        Standalone rollback arm
  run_framp*.sh          f-ramp (evidence growth) manipulation
  run_j2c.sh             J2c placebo control runs
  run_final.sh           Arms C and D (labelling, evidence separation)
  make_figures.py        Generates figures 1-5
  make_appendix.py       Generates appendix tables (LaTeX)
  analyze_loop.py        Closed-loop analysis (persistence, entry rates)
  final_audit.py         Number audit: every paper number traced to data

results/
  grid_*.jsonl           Static grid results (Table 2)
  loop_*.jsonl           Closed-loop results (Tables 4-8, Figures 3-5)

prompts/                 Judge prompt templates (J1, J2a, J2b, J2c)
stimuli/                 LoCoMo-derived test cells and manifest
figures/                 Paper figures (PDF)
data/processed/          Preprocessed LoCoMo data
```

## Data

Stimuli are derived from [LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., 2024).
