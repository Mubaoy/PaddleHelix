#!/usr/bin/env python3
"""Count GEM trainable/total parameters for the ChemVL protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parent
GEM_DIR = REPO_ROOT / "apps" / "pretrained_compound" / "ChemRL" / "GEM"
DEFAULT_OUTPUT = REPO_ROOT / "parameter_summary.csv"


ATOM_FEATURE_SIZES = {
    "atomic_num": 119,
    "formal_charge": 17,
    "degree": 12,
    "chiral_tag": 9,
    "total_numHs": 10,
    "is_aromatic": 2,
    "hybridization": 9,
}
BOND_FEATURE_SIZES = {"bond_dir": 7, "bond_type": 22, "is_in_ring": 2}
BOND_LENGTH_RBF_CENTERS = 20
BOND_ANGLE_RBF_CENTERS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PaddleHelix GEM parameters under the ChemVL setup.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="CSV output path.")
    parser.add_argument("--dataset", default="bbbp", help="Representative ChemVL dataset name.")
    parser.add_argument("--table", default="A", help="Representative ChemVL table label.")
    parser.add_argument("--split", default="scaffold", help="Representative ChemVL split name.")
    parser.add_argument("--task_type", choices=["classification", "regression"], default="classification")
    parser.add_argument("--num_tasks", type=int, default=1, help="Prediction-head output size.")
    parser.add_argument("--strict", action="store_true", help="Fail if direct model instantiation is unavailable.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_count(value: int | str | None) -> str:
    if value == "" or value is None:
        return ""
    value = int(value)
    if value >= 1_000_000:
        return "{:.2f}M".format(value / 1_000_000)
    if value >= 1_000:
        return "{:.2f}K".format(value / 1_000)
    return str(value)


def make_row(args: argparse.Namespace, trainable: int, total: int, status: str, notes: str) -> dict:
    return {
        "paper": "GEM",
        "variant": "GeoGNN/GEM",
        "finetune_strategy": "FT",
        "scope": "ChemVL Table {} {}, representative task {}".format(args.table, args.split, args.dataset),
        "task_alias": args.dataset,
        "num_tasks": args.num_tasks,
        "trainable": trainable,
        "total": total,
        "trainable_formatted": format_count(trainable),
        "total_formatted": format_count(total),
        "trainable_over_total": "{} / {}".format(format_count(trainable), format_count(total)),
        "status": status,
        "notes": notes,
    }


def mlp_parameter_count(layer_num: int, in_size: int, hidden_size: int, out_size: int) -> int:
    total = 0
    for layer_id in range(layer_num):
        if layer_id == 0:
            total += in_size * hidden_size + hidden_size
        elif layer_id < layer_num - 1:
            total += hidden_size * hidden_size + hidden_size
        else:
            total += hidden_size * out_size + out_size
    return total


def static_parameter_count(args: argparse.Namespace) -> tuple[int, int, str]:
    encoder_cfg = load_json(GEM_DIR / "model_configs" / "geognn_l8.json")
    head_cfg = load_json(GEM_DIR / "model_configs" / "down_mlp2.json")
    embed_dim = int(encoder_cfg.get("embed_dim", 32))
    layer_num = int(encoder_cfg.get("layer_num", 8))
    hidden_size = int(head_cfg.get("hidden_size", 128))
    head_layers = int(head_cfg.get("layer_num", 2))

    atom_embedding = sum((ATOM_FEATURE_SIZES[name] + 5) * embed_dim for name in encoder_cfg["atom_names"])
    bond_embedding = sum((BOND_FEATURE_SIZES[name] + 5) * embed_dim for name in encoder_cfg["bond_names"])
    bond_float_rbf = BOND_LENGTH_RBF_CENTERS * embed_dim + embed_dim
    bond_angle_float_rbf = BOND_ANGLE_RBF_CENTERS * embed_dim + embed_dim

    gin_mlp = mlp_parameter_count(2, embed_dim, embed_dim * 2, embed_dim)
    geognn_block = gin_mlp + 2 * embed_dim
    per_layer = bond_embedding + bond_float_rbf + bond_angle_float_rbf + 2 * geognn_block
    encoder_total = atom_embedding + bond_embedding + bond_float_rbf + layer_num * per_layer

    downstream_norm = 2 * embed_dim
    downstream_mlp = mlp_parameter_count(head_layers, embed_dim, hidden_size, args.num_tasks)
    total = encoder_total + downstream_norm + downstream_mlp
    detail = (
        "source=config/static; embed_dim={}; geognn_layers={}; head_layers={}; "
        "head_hidden={}; num_tasks={}".format(embed_dim, layer_num, head_layers, hidden_size, args.num_tasks)
    )
    return total, total, detail


def direct_parameter_count(args: argparse.Namespace) -> tuple[int, int, str]:
    for path in (REPO_ROOT, GEM_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import numpy as np
    from pahelix.model_zoo.gem_model import GeoGNNModel
    from pahelix.utils import load_json_config
    from src.model import DownstreamModel

    encoder_cfg = load_json_config(str(GEM_DIR / "model_configs" / "geognn_l8.json"))
    model_cfg = load_json_config(str(GEM_DIR / "model_configs" / "down_mlp2.json"))
    model_cfg["task_type"] = "class" if args.task_type == "classification" else "regr"
    model_cfg["num_tasks"] = args.num_tasks
    model = DownstreamModel(model_cfg, GeoGNNModel(encoder_cfg))
    params = list(model.parameters())
    total = int(sum(np.prod(param.shape) for param in params))
    trainable = int(sum(np.prod(param.shape) for param in params if not param.stop_gradient))
    return trainable, total, "source=model.parameters(); geognn_l8/down_mlp2; full fine-tuning"


def write_csv(row: dict, output: str) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper",
        "variant",
        "finetune_strategy",
        "scope",
        "task_alias",
        "num_tasks",
        "trainable",
        "total",
        "trainable_formatted",
        "total_formatted",
        "trainable_over_total",
        "status",
        "notes",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    return output_path


def main() -> None:
    args = parse_args()
    try:
        trainable, total, detail = direct_parameter_count(args)
        row = make_row(args, trainable, total, "OK", detail)
    except Exception as exc:
        if args.strict:
            raise
        trainable, total, detail = static_parameter_count(args)
        row = make_row(
            args,
            trainable,
            total,
            "STATIC_ARCHITECTURE_FALLBACK",
            "Direct GEM instantiation failed ({}: {}). {}.".format(type(exc).__name__, exc, detail),
        )

    output_path = write_csv(row, args.output)
    print("Wrote {}".format(output_path))
    print(json.dumps([row], indent=2))


if __name__ == "__main__":
    main()
