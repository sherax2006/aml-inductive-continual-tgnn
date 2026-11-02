# Inductive Continual Learning for Anti-Money Laundering Detection on Dynamic Transaction Graphs

Reference implementation for the paper *"Inductive Continual Learning for
Anti-Money Laundering Detection on Dynamic Transaction Graphs"* (Muhammad
Sheraz Rana, Ayyaz Hussain — Department of Computer Science, Quaid-i-Azam
University, Islamabad, Pakistan), submitted to *Expert Systems With
Applications*.

An inductive continual temporal graph neural network for AML detection on the
Elliptic Bitcoin transaction graph. The backbone samples localized *k*-hop
subgraphs (no computation is ever indexed by a node's identity), builds signed
per-edge messages, summarizes each neighbor relationship with an LSTM, and
aggregates neighbor summaries with GATv2 attention. Gradient Episodic Memory
(GEM) wraps the backbone as the continual-learning mechanism across a
seven-task temporal stream. Inductive generalization is validated directly,
not assumed: a held-out fraction of each task's nodes, and every edge
touching them, is removed from training entirely and restored only at
evaluation (the "cold-start" protocol — see §4.3 / §5.4 of the paper).

> **Citation:** this section will be filled in with the full citation and DOI
> once the paper is published. In the meantime, cite via the repository URL
> and the title/authors above.
>
> ```bibtex
> @article{rana2026inductive,
>   title   = {Inductive Continual Learning for Anti-Money Laundering Detection on Dynamic Transaction Graphs},
>   author  = {Rana, Muhammad Sheraz and Hussain, Dr. Ayyaz},
>   journal = {Expert Systems With Applications},
>   year    = {2026}
> }
> ```

## Repository layout

```
aml_tgnn/                      Core package
├── config.py                  Single ExperimentConfig dataclass -- every hyperparameter in Table 2
├── train.py                   Continual task-stream training entry point (python -m aml_tgnn.train)
├── infer.py                   Standalone inference over a saved checkpoint
├── task_matrix.py             ACC / BWT / FWT / ZS / FWT_std computation (Eqs. 1-5, Section 3.3)
├── models/
│   ├── message_backbone.py    Stages 1-4: subgraph sampling -> signed messages -> LSTM -> GATv2 (Section 4.1)
│   ├── classification_head.py Stage 5: shared MLP head (Section 4.1/4.2)
│   └── continual_wrapper.py   Wires the backbone + head + whichever continual-learning strategy is active
├── continual/
│   ├── gem.py                 Gradient Episodic Memory (Lopez-Paz & Ranzato, 2017) -- the paper's mechanism
│   ├── twp.py                 Topology-aware Weight Preserving (Liu et al., 2021) -- Section 5.6 ablation only
│   ├── ewc.py, agem.py, mas.py, replay_buffer.py   Additional strategies implemented but not reported in the paper (see "Scope" below)
├── data/
│   ├── elliptic_dataset.py    Elliptic loading + per-task temporal grouping
│   ├── coldstart.py           The held-out-node cold-start split (Section 4.3)
│   ├── split.py                Stratified per-task train/val(/cold-start) splitting
│   ├── temporal_loader.py     PyG NeighborLoader wiring for k-hop subgraph sampling
│   └── ibm_aml_dataset.py     A second, edge-classification data adapter -- not used for any result in the paper (see "Scope")
├── racla/                     An attempted RACLA reimplementation (Zhang et al., 2026) -- see "Scope"
└── utils/                     Metrics, checkpointing, run management, seeding

analysis_scripts/               Post-processing: raw checkpoints/logs -> the paper's exact table numbers
figure_scripts/                 Regenerates every figure in the paper (matplotlib, no external assets)
tests/test_smoke.py             Smoke-test suite covering the backbone, continual strategies, and data pipeline
```

### Scope — what this code does and doesn't reproduce

Every number reported in the paper comes from the **Elliptic** dataset,
the **`gem_twp`** continual-learning strategy (GEM, optionally combined with
TWP for the Section 5.6 ablation only), and **`node`**-level classification.
A few things exist in the code but are *not* behind any number in the paper:

- `continual/ewc.py`, `agem.py`, `mas.py`, `replay_buffer.py`, and the
  `ewc_replay` / `agem_twp` / `mas_twp` strategy choices — earlier/alternative
  continual-learning strategies this project explored. Keep them if you want
  to experiment further; `--continual-strategy gem_twp` is what reproduces the
  paper.
- `data/ibm_aml_dataset.py` and `analysis_scripts/preprocess_ibm_aml.py` — an
  edge-classification adapter for a second (IBM AML transactions) dataset.
  `train.py` imports it unconditionally, so it's kept for the code to run at
  all, but no result in the paper uses it.
- `aml_tgnn/racla/` — an independent reimplementation of RACLA (Zhang et al.,
  2026) attempted from its paper description alone (no public RACLA code
  exists). As the paper states in Section 5.5: *"A reimplementation built
  from the paper's description diverged substantially from its published
  figures, since the paper specifies neither its hyperparameters nor how its
  graph instance was constructed."* It's included for transparency, not as a
  validated baseline — Table 6's RACLA comparison uses RACLA's own published
  numbers, not this reimplementation's output.
- AMLWorld (Altman et al., 2024) is **not** in this repository at all —
  Section 5.7/5.8 of the paper discloses that extending evaluation to it is
  future work, blocked on GPU compute this study didn't have.

## Installation

```bash
git clone https://github.com/<your-username>/aml-inductive-continual-tgnn.git
cd aml-inductive-continual-tgnn
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

See `requirements.txt` for notes on installing `torch` / `torch_geometric`
for your hardware. **A GPU is not required** — every script auto-detects CUDA
and falls back to CPU (`aml_tgnn/config.py`'s `_default_device()`), and the
paper's own results were produced on CPU-only hardware.

## Data

The paper uses the publicly available **Elliptic** dataset:
<https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>. Download it
and point `--path` at the folder containing `elliptic_txs_features.csv`,
`elliptic_txs_classes.csv`, and `elliptic_txs_edgelist.csv` — e.g.:

```bash
elliptic_bitcoin_dataset/
├── elliptic_txs_features.csv
├── elliptic_txs_classes.csv
└── elliptic_txs_edgelist.csv
```

No new data were collected for this study; nothing else needs downloading.

## Quickstart: train the paper's main configuration

```bash
python -m aml_tgnn.train \
    --dataset elliptic --path elliptic_bitcoin_dataset \
    --time-steps-per-task 7 \
    --continual-strategy gem_twp --no-twp \
    --cold-start-ratio 0.15 \
    --seed 1
```

- `--time-steps-per-task 7` groups Elliptic's 49 raw timesteps into the
  paper's standard `T = 7` continual-learning tasks.
- `--continual-strategy gem_twp --no-twp` is GEM alone, wrapping the backbone
  — the architecture the paper proposes (Section 4.2). TWP is only ever
  switched on for the Section 5.6 ablation comparison (see below).
- `--cold-start-ratio 0.15` enables the held-out-node protocol (Table 2).
- Every other flag (learning rate, weight decay, epochs/task, GEM's memory
  budget/QP settings, batch size, neighbor-sampling fanout, dropout, attention
  heads) is a framework default — see `aml_tgnn/config.py` and Table 2 of the
  paper; none were hyperparameter-searched (Section 5.7 discloses this).

This writes per-task metrics to `Logs/Elliptic/<run_id>/task_metrics.csv` and
`cold_start_metrics.csv`, and checkpoints to
`Checkpoints/Elliptic/<run_id>/`. `<run_id>` defaults to a timestamp; pass
`--run-id <name>` to choose one (needed to match the naming the analysis
scripts below expect when reproducing the paper's exact multi-seed results).

To reproduce a result reported as mean ± std across five seeds, repeat the
run above for `--seed 1` through `--seed 5` (the train/validation/cold-start
split itself is held fixed across seeds — only initialization, dropout, and
sampling stochasticity vary, per Section 5.2).

## Reproducing the paper exactly

### Main results — Tables 3, 4, 6; Figures 4, 5, 6, 7

Train 5 seeds of the command above, naming each run
`my_gem_twp_component_gemonly_seed<N>` for `N` in `1..5`:

```bash
for seed in 1 2 3 4 5; do
  python -m aml_tgnn.train --dataset elliptic --path elliptic_bitcoin_dataset \
      --time-steps-per-task 7 --continual-strategy gem_twp --no-twp \
      --cold-start-ratio 0.15 --seed $seed \
      --run-id my_gem_twp_component_gemonly_seed$seed
done
```

`aml_tgnn/task_matrix.py`'s `compute_task_matrix` / `compute_summary_metrics`
turn those five runs' logs into ACC / BWT / FWT / ZS / FWT_std, mean ± std —
`analysis_scripts/compute_gem_twp_component_ablation.py` (below) shows exactly
how these are called against a directory of checkpoints.

### Ablation study — Table 7, Figure 8, Section 5.6

Four arms, train 5 seeds each (only 3 for "TWP only" — the paper discloses
this smaller sample explicitly):

| Arm (paper's label)         | Flags (in addition to the block above)   | Seeds | Run-id pattern                              |
|------------------------------|-------------------------------------------|-------|----------------------------------------------|
| GEM active (full architecture) | `--no-twp`                             | 1–5   | `my_gem_twp_component_gemonly_seed<N>`        |
| GEM + TWP                    | *(neither flag — both active)*            | 1–5   | `my_gem_twp_cs_seed<N>`                       |
| TWP only                     | `--no-gem`                                | 1–3   | `my_gem_twp_component_twponly_seed<N>`        |
| GEM inactive ("Neither")     | `--no-gem --no-twp`                       | 1–5   | `my_gem_twp_component_neither_seed<N>`        |

Then, from the repository root:

```bash
python3 analysis_scripts/compute_gem_twp_component_ablation.py
# -> analysis_scripts/gem_twp_component_ablation_results.json (Table 7's numbers)

python3 analysis_scripts/compute_ablation_significance.py
# -> the GEM-active-vs-GEM-inactive paired t-test reported in Section 5.6
#    (t(4)=13.83 / 13.89 / 10.47 / 10.76, all p<0.001)
```

`gem_twp_component_ablation_results.json` (already included in this repo —
reproducing it from scratch needs retraining, reading it does not) holds the
full per-seed values for all four arms, so the remaining significance tests
quoted in Section 5.6 — TWP-only vs. GEM-alone (paired), TWP-only vs. Neither
(independent), and GEM+TWP vs. GEM-alone (paired) — follow the same pattern
as `compute_ablation_significance.py`: load the two arms' `"values"` lists
and call `scipy.stats.ttest_rel` (matched seeds) or `ttest_ind` (independent
arms) directly.

Regenerate Figure 8 with:

```bash
python3 figure_scripts/figure_ablation.py
```

### Forward-transfer formulas — Figure 4, Section 5.5

```bash
python3 analysis_scripts/compute_fwt_variants.py
# -> analysis_scripts/fwt_variants_results.json
python3 figure_scripts/figure_fwt_formulas.py
```

### "Is graph structure doing anything?" — end of Section 5.6

```bash
python3 analysis_scripts/graph_necessity_and_disjoint_eval.py
# -> analysis_scripts/graph_necessity_results.json
# (edge-removal-at-test-time check + a random-forest-on-node-features-only baseline)
```

### Regenerating every figure

```bash
for f in figure_scripts/figure*.py; do python3 "$f"; done
```

Each script is self-contained and writes a `.png` to the current directory;
see each file's own docstring for the exact table/figure it corresponds to.

### Standalone inference on a trained checkpoint

```bash
python -m aml_tgnn.infer --dataset elliptic --path elliptic_bitcoin_dataset \
    --checkpoint Checkpoints/Elliptic/my_gem_twp_component_gemonly_seed1/task_7_complete.pt \
    --task-id 7
```

Restores the exact `ExperimentConfig` the checkpoint was trained with, so
there's no risk of mismatched flags silently producing wrong predictions.

## Tests

```bash
python3 -m pytest tests/test_smoke.py -v
```

Covers the backbone's five stages, each continual-learning strategy, the
cold-start split, and the metric formulas end to end on small synthetic
graphs (no real data needed).

## Hyperparameters

Every value in the paper's Table 2 lives in one place:
`aml_tgnn/config.py`'s `ExperimentConfig` (and its nested `BackboneConfig`,
`GEMConfig`, `TrainConfig`, `LoaderConfig` dataclasses). Nothing in this
codebase was hyperparameter-tuned against a validation set — Section 5.7 of
the paper states this explicitly, and the defaults here are exactly what
produced every reported number.

## License

MIT — see `LICENSE`. If you use this code, please cite the paper (see
"Citation" above).

## Authors

Muhammad Sheraz Rana (<sherazrana2006@gmail.com>) · Ayyaz Hussain
(<ayyaz.hussain@qau.edu.pk>) — Department of Computer Science, Quaid-i-Azam
University, Islamabad, Pakistan.
