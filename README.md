# PaddleHelix GEM under ChemVL Protocol

This repository is a research fork of PaddleHelix for running **ChemRL/GEM as an external baseline under the ChemVL MoleculeNet protocol**.

The goal is not to change the original GEM architecture. The goal is to fine-tune the official GEM pretrained checkpoints on the same ChemVL MoleculeNet datasets, split rules, seeds, metrics, and output layout used by our ChemVL comparison, so GEM can be compared fairly with ChemVL-style molecular representation methods.

The original PaddleHelix GEM code is under:

```text
apps/pretrained_compound/ChemRL/GEM/
```

The ChemVL reproduction adapter is under:

```text
apps/pretrained_compound/ChemRL/GEM/chemvl_protocol/
```

## What This Fork Adds

- ChemVL-compatible MoleculeNet fine-tuning runner for PaddleHelix ChemRL/GEM.
- `scaffold` and `random_scaffold` split logic aligned with the ChemVL protocol.
- Three-run protocol with `runseed = 1, 2, 3`.
- ChemVL-style outputs: `config.json`, `metrics.csv`, and `result.json`.
- Batch scripts for the two requested MoleculeNet tables.
- Result aggregation into mean +/- std CSV files and a summary plot.
- GEM 3D/MMFF feature caching to avoid rebuilding features for every run.

## Experiment Scope

This fork currently supports the MoleculeNet part of the comparison.

| Table | Split | Datasets |
|---|---|---|
| A | `scaffold` | BACE, BBBP, ClinTox, HIV, SIDER, Tox21, ESOL, FreeSolv, Lipo, QM7 |
| B | `random_scaffold` | BACE, BBBP, ClinTox, HIV, SIDER, Tox21, ESOL, FreeSolv, Lipo, QM7 |

MoleculeACE Table C is not implemented in this adapter yet. It needs a separate loader for the 30 MoleculeACE tasks and the MolMCL split protocol.

## Metrics

The metric convention follows the ChemVL main workflow.

| Task type | Dataset | Metric |
|---|---|---|
| Classification | BACE, BBBP, ClinTox, HIV, SIDER, Tox21 | ROC-AUC, higher is better |
| Regression | QM7 | MAE, lower is better |
| Regression | ESOL, FreeSolv, Lipo | RMSE, lower is better |

For every `(model, dataset, split)` setting, the batch scripts run:

```text
runseed = 1, 2, 3
```

For `scaffold`, the split is deterministic and `runseed` controls training randomness. For `random_scaffold`, `runseed` controls the random scaffold shuffle and the training randomness.

## Environment

The completed runs used this environment:

```bash
conda create -n paddlehelix_chemvl python=3.8 pip
conda activate paddlehelix_chemvl
pip install numpy==1.24.4 pandas scikit-learn networkx tqdm
pip install rdkit-pypi==2022.9.5 pgl==2.2.6 paddlepaddle-gpu==2.6.2
```

The RDKit version is important because the `scaffold` and `random_scaffold` splits depend on RDKit Murcko scaffold generation. For strict split reproducibility, use:

```text
rdkit-pypi==2022.9.5
```

The result analyzer also needs the ChemVL analysis script and plotting dependencies such as `matplotlib`. If the PaddleHelix environment does not include plotting packages, run `chemvl_protocol/analyze.sh` with another Python environment that has them.

## Required Data Layout

Set the ChemVL data root before running:

```bash
export CHEMVL_DATA_ROOT=/root/autodl-tmp/ChemVL-private/chemvl-data
```

`CHEMVL_DATA_ROOT` must contain the processed ChemVL MoleculeNet CSV files:

```text
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/classification/<task>/processed/<task>_processed_ac.csv
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/regression/<task>/processed/<task>_processed_ac.csv
```

The Lipo dataset is stored by ChemVL as `lipophilicity`; the runner accepts the short dataset id `lipo`.

## Pretrained GEM Checkpoints

Download the official PaddleHelix GEM pretrained checkpoints into the GEM directory:

```bash
cd apps/pretrained_compound/ChemRL/GEM
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/pretrained_models/compound/pretrain_models-chemrl_gem.tgz
tar xzf pretrain_models-chemrl_gem.tgz
```

The runner expects:

```text
apps/pretrained_compound/ChemRL/GEM/pretrain_models-chemrl_gem/class.pdparams
apps/pretrained_compound/ChemRL/GEM/pretrain_models-chemrl_gem/regr.pdparams
```

These checkpoints are runtime artifacts and should not be committed to the source repository.

## Run

Move to the GEM directory first:

```bash
cd apps/pretrained_compound/ChemRL/GEM
```

Check the split sizes without training:

```bash
SPLIT_ONLY=1 RUNSEED_START=1 RUNSEED_END=1 bash chemvl_protocol/run_moleculenet_scaffold.sh
SPLIT_ONLY=1 RUNSEED_START=1 RUNSEED_END=1 bash chemvl_protocol/run_moleculenet_random_scaffold.sh
```

Dry-run the commands:

```bash
DRY_RUN=1 bash chemvl_protocol/run_moleculenet_scaffold.sh
DRY_RUN=1 bash chemvl_protocol/run_moleculenet_random_scaffold.sh
```

Run Table A:

```bash
bash chemvl_protocol/run_moleculenet_scaffold.sh
```

Run Table B:

```bash
bash chemvl_protocol/run_moleculenet_random_scaffold.sh
```

Run both sequentially in the background:

```bash
bash chemvl_protocol/run_ab_background.sh
```

The background script prints a pid file and a log file, for example:

```text
ab pidfile=/root/autodl-tmp/ChemVL-private/chemvl-data/results/moleculenet/gem_under_chemvl_logs/ab_<timestamp>.pid log=/root/autodl-tmp/ChemVL-private/chemvl-data/results/moleculenet/gem_under_chemvl_logs/ab_<timestamp>.log
```

Useful overrides:

```bash
PYTHON=/path/to/python RUNSEED_START=1 RUNSEED_END=3 MAX_EPOCH=100 \
  bash chemvl_protocol/run_moleculenet_scaffold.sh
```

By default, completed runs are skipped if a matching `result.json` already exists. Set `NO_SKIP=1` to force reruns.

## Aggregate Results

After all runs finish:

```bash
PYTHON=/path/to/python-with-chemvl-and-matplotlib bash chemvl_protocol/analyze.sh
```

Outputs are written under:

```text
${CHEMVL_DATA_ROOT}/results/moleculenet/gem_under_chemvl/
```

Important files:

```text
gem_under_chemvl_summary_by_dataset.csv
gem_under_chemvl_summary_macro.csv
gem_under_chemvl_summary.png
```

Each individual run is stored as:

```text
<result_root>/<version>/<dataset>/<timestamp_seed>/
|-- config.json
|-- metrics.csv
`-- result.json
```

The GEM feature cache is stored under:

```text
${CHEMVL_DATA_ROOT}/cache/gem_under_chemvl/<dataset>/
```

## Verified Split Sizes

The following sizes were checked with `RUNSEED_START=1 RUNSEED_END=1 SPLIT_ONLY=1`.

| Dataset | `scaffold` train/valid/test | `random_scaffold` train/valid/test |
|---|---:|---:|
| bace | 1210/151/152 | 1211/151/151 |
| bbbp | 1631/204/204 | 1633/203/203 |
| clintox | 1182/148/148 | 1184/147/147 |
| hiv | 32901/4113/4113 | 32903/4112/4112 |
| sider | 1141/143/143 | 1143/142/142 |
| tox21 | 6264/783/784 | 6265/783/783 |
| esol | 902/113/113 | 904/112/112 |
| freesolv | 513/64/65 | 514/64/64 |
| lipo | 3360/420/420 | 3360/420/420 |
| qm7 | 5464/685/681 | 5464/683/683 |

## Completed A/B Results

The table below reports mean +/- std over three runs. Classification numbers are ROC-AUC. Regression numbers are RMSE except QM7, which is MAE.

| Split | Dataset | Metric | Mean +/- std |
|---|---|---|---:|
| `scaffold` | bace | ROC-AUC | 0.8065 +/- 0.0051 |
| `scaffold` | bbbp | ROC-AUC | 0.6916 +/- 0.0170 |
| `scaffold` | clintox | ROC-AUC | 0.8710 +/- 0.0240 |
| `scaffold` | hiv | ROC-AUC | 0.7766 +/- 0.0111 |
| `scaffold` | sider | ROC-AUC | 0.6071 +/- 0.0329 |
| `scaffold` | tox21 | ROC-AUC | 0.7748 +/- 0.0055 |
| `scaffold` | esol | RMSE | 0.8216 +/- 0.0397 |
| `scaffold` | freesolv | RMSE | 1.9541 +/- 0.2165 |
| `scaffold` | lipo | RMSE | 0.6795 +/- 0.0122 |
| `scaffold` | qm7 | MAE | 74.4755 +/- 2.0553 |
| `random_scaffold` | bace | ROC-AUC | 0.8587 +/- 0.0437 |
| `random_scaffold` | bbbp | ROC-AUC | 0.8479 +/- 0.0664 |
| `random_scaffold` | clintox | ROC-AUC | 0.8575 +/- 0.0448 |
| `random_scaffold` | hiv | ROC-AUC | 0.7686 +/- 0.0352 |
| `random_scaffold` | sider | ROC-AUC | 0.6378 +/- 0.0168 |
| `random_scaffold` | tox21 | ROC-AUC | 0.8180 +/- 0.0090 |
| `random_scaffold` | esol | RMSE | 0.8817 +/- 0.0998 |
| `random_scaffold` | freesolv | RMSE | 2.1247 +/- 0.4496 |
| `random_scaffold` | lipo | RMSE | 0.6101 +/- 0.0392 |
| `random_scaffold` | qm7 | MAE | 53.7108 +/- 16.9924 |

## Code Map

| File | Purpose |
|---|---|
| `apps/pretrained_compound/ChemRL/GEM/chemvl_protocol/finetune_gem.py` | Single-run fine-tuning entry point |
| `apps/pretrained_compound/ChemRL/GEM/chemvl_protocol/batch_run.py` | Expands dataset x runseed jobs and skips completed runs |
| `apps/pretrained_compound/ChemRL/GEM/chemvl_protocol/run_moleculenet_scaffold.sh` | Table A runner |
| `apps/pretrained_compound/ChemRL/GEM/chemvl_protocol/run_moleculenet_random_scaffold.sh` | Table B runner |
| `apps/pretrained_compound/ChemRL/GEM/chemvl_protocol/run_ab_background.sh` | Runs Table A and Table B in the background |
| `apps/pretrained_compound/ChemRL/GEM/chemvl_protocol/analyze.sh` | Aggregates ChemVL-style `result.json` files |
| `apps/pretrained_compound/ChemRL/GEM/chemvl_protocol/dataset_list_moleculenet_cls6.txt` | Six classification datasets |
| `apps/pretrained_compound/ChemRL/GEM/chemvl_protocol/dataset_list_moleculenet_reg4.txt` | Four regression datasets |

## Reproducibility Notes

- The split implementation uses RDKit Murcko scaffolds with chirality enabled by default.
- Classification labels are converted from ChemVL's `0/1/-1 missing` convention to GEM's `-1/1/0 missing` convention before training.
- Regression labels are standardized using the training split by default (`--label-stat-scope train`).
- GEM 3D/MMFF preprocessing is CPU-heavy, especially for HIV. The cache is reused across splits and seeds unless `--refresh-cache` is set.
- With PaddlePaddle 2.6.2 and the current PaddleHelix GEM config, loading the official GEM checkpoint may print warnings that two atom embedding tensors are skipped because of shape mismatches. The completed tables above were generated under that setup.

## Original PaddleHelix/GEM

This repository is based on PaddleHelix and its ChemRL/GEM implementation:

- PaddleHelix: <https://github.com/PaddlePaddle/PaddleHelix>
- GEM code path: `apps/pretrained_compound/ChemRL/GEM`
- GEM paper: <https://www.nature.com/articles/s42256-021-00438-4>

Please cite the original GEM work when using the GEM model or pretrained checkpoints:

```bibtex
@article{fang2022geometry,
  title={Geometry-enhanced molecular representation learning for property prediction},
  author={Fang, Xiaomin and Liu, Lihang and Lei, Jieqiong and He, Donglong and Zhang, Shanzhuo and Zhou, Jingbo and Wang, Fan and Wu, Hua and Wang, Haifeng},
  journal={Nature Machine Intelligence},
  pages={1--8},
  year={2022},
  publisher={Nature Publishing Group},
  doi={10.1038/s42256-021-00438-4}
}
```

## License

The upstream PaddleHelix project is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. Keep the upstream license terms when redistributing this fork.
