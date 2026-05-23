#!/usr/bin/env python3
"""Batch driver for PaddleHelix GEM under ChemVL MoleculeNet splits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent

CLS6 = ["bace", "bbbp", "clintox", "hiv", "sider", "tox21"]
REG4 = ["esol", "freesolv", "lipo", "qm7"]


def load_dataset_list(path: Optional[str], fallback: List[str]) -> List[str]:
    if path is None:
        return fallback
    out: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def infer_task_type(datasets: List[str]) -> str:
    if all(d in CLS6 for d in datasets):
        return "classification"
    if all(d in REG4 for d in datasets):
        return "regression"
    raise ValueError(f"Mixed or unsupported datasets: {datasets}")


def output_has_result(output_root: Path, version: str, dataset: str, runseed: int, split: str) -> bool:
    ddir = output_root / version / dataset
    if not ddir.is_dir():
        return False
    for run_dir in ddir.iterdir():
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        res_path = run_dir / "result.json"
        if not cfg_path.is_file() or not res_path.is_file():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            res = json.loads(res_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if cfg.get("training", {}).get("runseed") == runseed and cfg.get("dataset", {}).get("split") == split:
            if res.get("best_valid_on_test") is not None:
                return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["scaffold", "random_scaffold"], required=True)
    p.add_argument("--dataset-list", default=None)
    p.add_argument("--datasets", default="")
    p.add_argument("--task-type", choices=["classification", "regression"], default=None)
    p.add_argument("--runseed-start", type=int, default=1)
    p.add_argument("--runseed-end", type=int, default=3)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--finetune-script", default=str(_SCRIPT_DIR / "finetune_gem.py"))
    p.add_argument("--chemvl-data-root", default=os.environ.get("CHEMVL_DATA_ROOT", "/root/autodl-tmp/ChemVL-private/chemvl-data"))
    p.add_argument("--output-root", default=None)
    p.add_argument("--exp-name", default="gem_under_chemvl")
    p.add_argument("--max-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--loader-workers", type=int, default=1)
    p.add_argument("--encoder-lr", type=float, default=1e-3)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--dropout-rate", type=float, default=0.2)
    p.add_argument("--split-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--capture-output", action="store_true")
    args = p.parse_args()

    chemvl_data_root = Path(args.chemvl_data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else chemvl_data_root / "results" / "moleculenet" / args.exp_name

    if args.dataset_list:
        datasets = load_dataset_list(args.dataset_list, [])
    elif args.datasets:
        datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = CLS6 + REG4
    if not datasets:
        print("No datasets provided.", file=sys.stderr)
        return 1
    task_type = args.task_type or infer_task_type(datasets)
    version = f"gem_moleculenet_{'cls' if task_type == 'classification' else 'reg'}_{args.split.replace('_', '-')}"

    failures = 0
    for dataset in datasets:
        for runseed in range(args.runseed_start, args.runseed_end + 1):
            if not args.no_skip_existing and not args.split_only and output_has_result(output_root, version, dataset, runseed, args.split):
                print(f"Skip {dataset} / runseed={runseed} (existing result.json)")
                continue
            cmd = [
                args.python,
                args.finetune_script,
                "--dataset", dataset,
                "--task-type", task_type,
                "--split", args.split,
                "--runseed", str(runseed),
                "--chemvl-data-root", str(chemvl_data_root),
                "--output-root", str(output_root),
                "--version", version,
                "--max-epoch", str(args.max_epoch),
                "--num-workers", str(args.num_workers),
                "--loader-workers", str(args.loader_workers),
                "--encoder-lr", str(args.encoder_lr),
                "--head-lr", str(args.head_lr),
                "--dropout-rate", str(args.dropout_rate),
            ]
            if args.batch_size is not None:
                cmd.extend(["--batch-size", str(args.batch_size)])
            if args.split_only:
                cmd.append("--split-only")
            print("Executing:", " ".join(cmd))
            if args.dry_run:
                continue
            result = subprocess.run(cmd, capture_output=args.capture_output, text=True)
            if args.capture_output:
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
            if result.returncode != 0:
                failures += 1
                print(f"Failed {dataset} / runseed={runseed}: exit {result.returncode}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
