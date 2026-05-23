# GEM ChemVL Protocol

This directory contains the adapter used to run PaddleHelix ChemRL/GEM on the ChemVL MoleculeNet A/B protocol.

The adapter keeps GEM's model code unchanged and provides ChemVL-compatible data loading, splitting, training, output files, and aggregation.

## Files

| File | Purpose |
|---|---|
| `finetune_gem.py` | Single dataset/split/runseed fine-tuning entry point |
| `batch_run.py` | Expands datasets and runseeds into fine-tuning jobs |
| `run_moleculenet_scaffold.sh` | Table A runner for the `scaffold` split |
| `run_moleculenet_random_scaffold.sh` | Table B runner for the `random_scaffold` split |
| `run_ab_background.sh` | Runs Table A then Table B in a detached background process |
| `analyze.sh` | Aggregates ChemVL-style `result.json` files |
| `dataset_list_moleculenet_cls6.txt` | BACE, BBBP, ClinTox, HIV, SIDER, Tox21 |
| `dataset_list_moleculenet_reg4.txt` | ESOL, FreeSolv, Lipo, QM7 |

## Quick Start

From `apps/pretrained_compound/ChemRL/GEM`:

```bash
export CHEMVL_DATA_ROOT=/root/autodl-tmp/ChemVL-private/chemvl-data
```

Check split sizes only:

```bash
SPLIT_ONLY=1 RUNSEED_START=1 RUNSEED_END=1 bash chemvl_protocol/run_moleculenet_scaffold.sh
SPLIT_ONLY=1 RUNSEED_START=1 RUNSEED_END=1 bash chemvl_protocol/run_moleculenet_random_scaffold.sh
```

Run the two requested tables:

```bash
bash chemvl_protocol/run_moleculenet_scaffold.sh
bash chemvl_protocol/run_moleculenet_random_scaffold.sh
```

Or run both in the background:

```bash
bash chemvl_protocol/run_ab_background.sh
```

Aggregate results:

```bash
PYTHON=/path/to/python-with-chemvl-and-matplotlib bash chemvl_protocol/analyze.sh
```

## Defaults

- Python: `/root/miniconda3/envs/paddlehelix_chemvl/bin/python`
- Epochs: `MAX_EPOCH=100`
- Runseeds: `RUNSEED_START=1`, `RUNSEED_END=3`
- RDKit feature workers: `NUM_WORKERS=16`
- PGL loader workers: `LOADER_WORKERS=1`
- OpenMP threads: `GEM_OMP_NUM_THREADS=1`
- Output root: `${CHEMVL_DATA_ROOT}/results/moleculenet/gem_under_chemvl`
- Feature cache: `${CHEMVL_DATA_ROOT}/cache/gem_under_chemvl`

Useful overrides:

```bash
PYTHON=/path/to/python RUNSEED_START=2 RUNSEED_END=3 MAX_EPOCH=100 \
  bash chemvl_protocol/run_moleculenet_scaffold.sh
```

Completed jobs are skipped when a matching `result.json` exists. Use `NO_SKIP=1` to force reruns.
