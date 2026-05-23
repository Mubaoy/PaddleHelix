#!/usr/bin/env python3
"""Fine-tune PaddleHelix GEM under ChemVL MoleculeNet split protocols."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score

_SCRIPT_DIR = Path(__file__).resolve().parent
_GEM_DIR = _SCRIPT_DIR.parent
_PADDLEHELIX_ROOT = _SCRIPT_DIR.parents[4]
for _path in (str(_PADDLEHELIX_ROOT), str(_GEM_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pahelix.datasets.inmemory_dataset import InMemoryDataset
from pahelix.model_zoo.gem_model import GeoGNNModel
from pahelix.utils import load_json_config
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_MMFF3d
from src.featurizer import DownstreamCollateFn
from src.model import DownstreamModel

CLASSIFICATION_DATASETS = {"bace", "bbbp", "clintox", "hiv", "sider", "tox21"}
REGRESSION_DATASETS = {"esol", "freesolv", "lipo", "qm7"}
REGRESSION_STEM = {"lipo": "lipophilicity"}


def default_chemvl_data_root() -> Path:
    return Path(os.environ.get("CHEMVL_DATA_ROOT", "/root/autodl-tmp/ChemVL-private/chemvl-data")).resolve()


def infer_task_type(dataset: str) -> str:
    if dataset in CLASSIFICATION_DATASETS:
        return "classification"
    if dataset in REGRESSION_DATASETS:
        return "regression"
    raise ValueError(f"Unsupported MoleculeNet dataset: {dataset}")


def processed_csv_path(data_root: Path, dataset: str, task_type: str) -> Path:
    split_dir = "classification" if task_type == "classification" else "regression"
    stem = REGRESSION_STEM.get(dataset, dataset)
    return data_root / "finetuning_datasets" / "MPP" / split_dir / stem / "processed" / f"{stem}_processed_ac.csv"


def parse_processed_ac(path: Path, task_type: str) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path)
    smiles = df["smiles"].astype(str).tolist()
    labels = np.array(df["label"].apply(lambda x: str(x).split()).tolist())
    if task_type == "classification":
        labels = labels.astype(np.int64)
    else:
        labels = labels.astype(np.float32)
    return smiles, labels


def chemvl_to_gem_class_labels(labels: np.ndarray) -> np.ndarray:
    # ChemVL uses 0/1 labels and -1 for missing multitask labels.
    # GEM collate expects -1/1 labels and 0 for missing labels.
    labels = np.asarray(labels, dtype=np.int64)
    return np.where(labels < 0, 0, np.where(labels == 0, -1, 1)).astype(np.float32)


def generate_scaffold(smiles: str, include_chirality: bool = True) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)


def scaffold_split(
    smiles_list: Sequence[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    include_chirality: bool = True,
) -> Tuple[List[int], List[int], List[int]]:
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    all_scaffolds: Dict[str, List[int]] = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=include_chirality)
        all_scaffolds.setdefault(scaffold, []).append(i)
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set
        for _, scaffold_set in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)
    return train_idx, valid_idx, test_idx


def random_scaffold_split(
    smiles_list: Sequence[str],
    seed: int,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    include_chirality: bool = True,
) -> Tuple[List[int], List[int], List[int]]:
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    rng = np.random.RandomState(seed)
    scaffolds: Dict[str, List[int]] = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=include_chirality)
        scaffolds[scaffold].append(ind)
    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))
    n_total_valid = int(np.floor(frac_valid * len(smiles_list)))
    n_total_test = int(np.floor(frac_test * len(smiles_list)))
    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []
    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)
    return train_idx, valid_idx, test_idx


def get_split(args: argparse.Namespace, smiles: Sequence[str]) -> Tuple[List[int], List[int], List[int]]:
    if args.split == "scaffold":
        return scaffold_split(smiles, include_chirality=args.chirality)
    if args.split == "random_scaffold":
        return random_scaffold_split(smiles, seed=args.runseed, include_chirality=args.chirality)
    raise ValueError(args.split)


class QuietDownstreamTransformFn:
    def __init__(self, task_type: str):
        self.task_type = task_type

    def __call__(self, raw_data: Dict[str, Any]):
        mol = AllChem.MolFromSmiles(raw_data["smiles"])
        if mol is None:
            return None
        try:
            data = mol_to_geognn_graph_data_MMFF3d(mol)
        except Exception as exc:  # RDKit conformer generation can fail on rare molecules.
            print(f"Drop invalid 3D molecule at index {raw_data['orig_index']}: {exc}", file=sys.stderr)
            return None
        data["label"] = raw_data["label"].reshape([-1])
        data["orig_index"] = np.array([raw_data["orig_index"]], dtype=np.int64)
        return data


def cache_has_npz(path: Path) -> bool:
    return path.is_dir() and any(p.suffix == ".npz" for p in path.iterdir())


def build_or_load_features(
    smiles: Sequence[str],
    labels_for_gem: np.ndarray,
    task_type: str,
    cache_path: Path,
    num_workers: int,
    refresh_cache: bool,
) -> InMemoryDataset:
    if refresh_cache and cache_path.exists():
        shutil.rmtree(cache_path)
    if cache_has_npz(cache_path):
        print(f"Read cached GEM features: {cache_path}")
        return InMemoryDataset(npz_data_path=str(cache_path))

    print(f"Build GEM 3D features: {cache_path}")
    raw = [
        {
            "smiles": smi,
            "label": np.asarray(labels_for_gem[i]),
            "orig_index": i,
        }
        for i, smi in enumerate(smiles)
    ]
    dataset = InMemoryDataset(data_list=raw)
    dataset.transform(QuietDownstreamTransformFn(task_type), num_workers=num_workers, drop_none=True)
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset.save_data(str(cache_path))
    return dataset


def subset_from_orig_indices(dataset: InMemoryDataset, indices: Sequence[int], split_name: str) -> InMemoryDataset:
    by_orig: Dict[int, Dict[str, Any]] = {}
    for item in dataset.data_list:
        orig = int(np.asarray(item["orig_index"]).reshape(-1)[0])
        by_orig[orig] = item
    missing = [int(i) for i in indices if int(i) not in by_orig]
    if missing:
        print(f"{split_name}: dropped {len(missing)} molecules without GEM features", file=sys.stderr)
    return InMemoryDataset(data_list=[by_orig[int(i)] for i in indices if int(i) in by_orig])


def calc_rocauc_score(labels: np.ndarray, preds: np.ndarray, valid: np.ndarray) -> float:
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
        preds = preds.reshape(-1, 1)
    rocs = []
    for i in range(labels.shape[1]):
        c_valid = valid[:, i].astype(bool)
        c_label = labels[c_valid, i]
        c_pred = preds[c_valid, i]
        if len(np.unique(c_label)) == 2:
            rocs.append(roc_auc_score(c_label, c_pred))
    if not rocs:
        raise RuntimeError("No valid binary task for ROC-AUC.")
    return float(np.mean(rocs))


def calc_rmse(labels: np.ndarray, preds: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - labels) ** 2)))


def calc_mae(labels: np.ndarray, preds: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - labels)))


def exempt_parameters(src_list, ref_list):
    out = []
    for x in src_list:
        if not any(x is y for y in ref_list):
            out.append(x)
    return out


def train_class(args, model, train_dataset, collate_fn, criterion, encoder_opt, head_opt) -> float:
    data_gen = train_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )
    losses = []
    model.train()
    for atom_bond_graphs, bond_angle_graphs, valids, labels in data_gen:
        if len(labels) < args.batch_size * 0.5:
            continue
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        labels = paddle.to_tensor(labels, "float32")
        valids = paddle.to_tensor(valids, "float32")
        preds = model(atom_bond_graphs, bond_angle_graphs)
        loss = criterion(preds, labels)
        denom = paddle.sum(valids)
        if float(denom.numpy()) <= 0:
            continue
        loss = paddle.sum(loss * valids) / denom
        loss.backward()
        encoder_opt.step()
        head_opt.step()
        encoder_opt.clear_grad()
        head_opt.clear_grad()
        losses.append(float(loss.numpy()))
    return float(np.mean(losses)) if losses else float("nan")


def eval_class(args, model, dataset, collate_fn) -> float:
    data_gen = dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    total_pred, total_label, total_valid = [], [], []
    model.eval()
    for atom_bond_graphs, bond_angle_graphs, valids, labels in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        preds = model(atom_bond_graphs, bond_angle_graphs)
        total_pred.append(preds.numpy())
        total_valid.append(np.asarray(valids))
        total_label.append(np.asarray(labels))
    return calc_rocauc_score(np.concatenate(total_label, 0), np.concatenate(total_pred, 0), np.concatenate(total_valid, 0))


def train_regr(args, model, label_mean, label_std, train_dataset, collate_fn, criterion, encoder_opt, head_opt) -> float:
    data_gen = train_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )
    losses = []
    model.train()
    for atom_bond_graphs, bond_angle_graphs, labels in data_gen:
        if len(labels) < args.batch_size * 0.5:
            continue
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        scaled_labels = (np.asarray(labels) - label_mean) / (label_std + 1e-5)
        scaled_labels = paddle.to_tensor(scaled_labels, "float32")
        preds = model(atom_bond_graphs, bond_angle_graphs)
        loss = criterion(preds, scaled_labels)
        loss.backward()
        encoder_opt.step()
        head_opt.step()
        encoder_opt.clear_grad()
        head_opt.clear_grad()
        losses.append(float(loss.numpy()))
    return float(np.mean(losses)) if losses else float("nan")


def eval_regr(args, model, label_mean, label_std, dataset, collate_fn, metric_name: str) -> float:
    data_gen = dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    total_pred, total_label = [], []
    model.eval()
    for atom_bond_graphs, bond_angle_graphs, labels in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        scaled_preds = model(atom_bond_graphs, bond_angle_graphs)
        preds = scaled_preds.numpy() * label_std + label_mean
        total_pred.append(preds)
        total_label.append(np.asarray(labels))
    labels = np.concatenate(total_label, 0)
    preds = np.concatenate(total_pred, 0)
    if metric_name == "mae":
        return calc_mae(labels, preds)
    return calc_rmse(labels, preds)


def is_better(left: float, right: float, mode: str) -> bool:
    if mode == "max":
        return left > right
    if mode == "min":
        return left < right
    raise ValueError(mode)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def make_output_dir(args, task_type: str) -> Path:
    output_root = Path(args.output_root).expanduser().resolve()
    version = args.version or f"gem_moleculenet_{'cls' if task_type == 'classification' else 'reg'}_{args.split.replace('_', '-') }"
    stamp = datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d_%H_%M_%S") + f"_seed{args.runseed}"
    out = output_root / version / args.dataset / stamp
    out.mkdir(parents=True, exist_ok=False)
    return out


def build_config(args, task_type: str, source_csv: Path, split_sizes: Dict[str, int], version: str) -> Dict[str, Any]:
    return {
        "basic": {
            "version": version,
            "log_dir_base": str(Path(args.output_root).expanduser().resolve().parent),
            "exp_name": Path(args.output_root).name,
            "gpu": args.device,
            "num_workers": args.num_workers,
            "description": "PaddleHelix GEM baseline under ChemVL MoleculeNet split protocol.",
        },
        "dataset": {
            "benchmark": "moleculenet",
            "dataset": args.dataset,
            "source_csv": str(source_csv),
            "split": args.split,
            "task_type": task_type,
            "chirality": args.chirality,
            "num_tasks": split_sizes["num_tasks"],
            "train_size": split_sizes["train"],
            "valid_size": split_sizes["valid"],
            "test_size": split_sizes["test"],
        },
        "model": {
            "name": "PaddleHelix ChemRL/GEM",
            "compound_encoder_config": str(Path(args.compound_encoder_config).resolve()),
            "model_config": str(Path(args.model_config).resolve()),
            "init_model": str(Path(args.init_model).resolve()),
            "dropout_rate": args.dropout_rate,
        },
        "training": {
            "seed": args.runseed,
            "runseed": args.runseed,
            "epochs": args.max_epoch,
            "batch_size": args.batch_size,
            "encoder_lr": args.encoder_lr,
            "head_lr": args.head_lr,
            "label_stat_scope": args.label_stat_scope,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=["scaffold", "random_scaffold"], required=True)
    parser.add_argument("--runseed", type=int, default=1)
    parser.add_argument("--task-type", choices=["classification", "regression"], default=None)
    parser.add_argument("--chemvl-data-root", default=str(default_chemvl_data_root()))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--compound-encoder-config", default=str(_GEM_DIR / "model_configs" / "geognn_l8.json"))
    parser.add_argument("--model-config", default=str(_GEM_DIR / "model_configs" / "down_mlp2.json"))
    parser.add_argument("--init-model", default=None)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4, help="workers for RDKit feature preprocessing")
    parser.add_argument("--loader-workers", type=int, default=1, help="workers for PGL data loading")
    parser.add_argument("--encoder-lr", type=float, default=1e-3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--label-stat-scope", choices=["train", "full"], default="train")
    parser.add_argument("--no-chirality", dest="chirality", action="store_false")
    parser.set_defaults(chirality=True)
    parser.add_argument("--split-only", action="store_true")
    args = parser.parse_args()

    args.dataset = args.dataset.strip().lower()
    task_type = args.task_type or infer_task_type(args.dataset)
    data_root = Path(args.chemvl_data_root).expanduser().resolve()
    if args.output_root is None:
        args.output_root = str(data_root / "results" / "moleculenet" / "gem_under_chemvl")
    if args.cache_root is None:
        args.cache_root = str(data_root / "cache" / "gem_under_chemvl")
    if args.init_model is None:
        ckpt_name = "class.pdparams" if task_type == "classification" else "regr.pdparams"
        args.init_model = str(_GEM_DIR / "pretrain_models-chemrl_gem" / ckpt_name)
    if args.batch_size is None:
        if args.dataset == "tox21":
            args.batch_size = 128
        elif args.dataset == "freesolv":
            args.batch_size = 30
        else:
            args.batch_size = 32

    source_csv = processed_csv_path(data_root, args.dataset, task_type)
    if not source_csv.is_file():
        raise FileNotFoundError(f"ChemVL processed CSV not found: {source_csv}")
    if not Path(args.init_model).is_file():
        raise FileNotFoundError(f"GEM init model not found: {args.init_model}")

    set_seed(args.runseed)
    smiles, chemvl_labels = parse_processed_ac(source_csv, task_type)
    train_idx, valid_idx, test_idx = get_split(args, smiles)
    num_tasks = int(chemvl_labels.shape[1]) if chemvl_labels.ndim > 1 else 1
    split_sizes = {"train": len(train_idx), "valid": len(valid_idx), "test": len(test_idx), "num_tasks": num_tasks}
    print(
        f"Split {args.dataset}/{args.split}/seed={args.runseed}: "
        f"train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)} num_tasks={num_tasks}"
    )
    if args.split_only:
        return 0

    if paddle.device.is_compiled_with_cuda() and args.device.startswith("gpu"):
        paddle.set_device(args.device)
    else:
        paddle.set_device("cpu")

    labels_for_gem = chemvl_to_gem_class_labels(chemvl_labels) if task_type == "classification" else chemvl_labels.astype(np.float32)
    cache_path = Path(args.cache_root).expanduser().resolve() / args.dataset
    full_dataset = build_or_load_features(
        smiles,
        labels_for_gem,
        task_type,
        cache_path,
        num_workers=args.num_workers,
        refresh_cache=args.refresh_cache,
    )
    train_dataset = subset_from_orig_indices(full_dataset, train_idx, "train")
    valid_dataset = subset_from_orig_indices(full_dataset, valid_idx, "valid")
    test_dataset = subset_from_orig_indices(full_dataset, test_idx, "test")
    split_sizes.update({"train": len(train_dataset), "valid": len(valid_dataset), "test": len(test_dataset)})
    print(f"GEM feature split sizes: {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}")

    compound_encoder_config = load_json_config(args.compound_encoder_config)
    compound_encoder_config["dropout_rate"] = args.dropout_rate
    model_config = load_json_config(args.model_config)
    model_config["dropout_rate"] = args.dropout_rate
    model_config["task_type"] = "class" if task_type == "classification" else "regr"
    model_config["num_tasks"] = num_tasks

    compound_encoder = GeoGNNModel(compound_encoder_config)
    model = DownstreamModel(model_config, compound_encoder)
    compound_encoder.set_state_dict(paddle.load(args.init_model))
    print(f"Load state_dict from {args.init_model}")

    encoder_params = compound_encoder.parameters()
    head_params = exempt_parameters(model.parameters(), encoder_params)
    encoder_opt = paddle.optimizer.Adam(args.encoder_lr, parameters=encoder_params)
    head_opt = paddle.optimizer.Adam(args.head_lr, parameters=head_params)
    collate_fn = DownstreamCollateFn(
        atom_names=compound_encoder_config["atom_names"],
        bond_names=compound_encoder_config["bond_names"],
        bond_float_names=compound_encoder_config["bond_float_names"],
        bond_angle_float_names=compound_encoder_config["bond_angle_float_names"],
        task_type=model_config["task_type"],
    )

    out_dir = make_output_dir(args, task_type)
    version = out_dir.parents[1].name
    cfg = build_config(args, task_type, source_csv, split_sizes, version)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=4), encoding="utf-8")

    if task_type == "classification":
        criterion = nn.BCELoss(reduction="none")
        metric_name = "rocauc"
        select_mode = "max"
        best_valid = -float("inf")
    else:
        criterion = nn.L1Loss()
        metric_name = "mae" if args.dataset == "qm7" else "rmse"
        select_mode = "min"
        best_valid = float("inf")
        stat_dataset = train_dataset if args.label_stat_scope == "train" else full_dataset
        stat_labels = np.array([data["label"] for data in stat_dataset], dtype=np.float32)
        label_mean = np.reshape(np.mean(stat_labels, axis=0), [1, -1])
        label_std = np.reshape(np.std(stat_labels, axis=0), [1, -1])
        print(f"label_mean={label_mean.flatten().tolist()} label_std={label_std.flatten().tolist()}")

    best_valid_on_test = None
    best_valid_epoch = -1
    best_train_loss = float("inf")
    best_train_epoch = -1
    best_train_on_test = None
    history_path = out_dir / "metrics.csv"
    with history_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "valid", "test"])
        writer.writeheader()
        for epoch in range(args.max_epoch):
            if task_type == "classification":
                train_loss = train_class(args, model, train_dataset, collate_fn, criterion, encoder_opt, head_opt)
                valid_metric = eval_class(args, model, valid_dataset, collate_fn)
                test_metric = eval_class(args, model, test_dataset, collate_fn)
            else:
                train_loss = train_regr(
                    args, model, label_mean, label_std, train_dataset, collate_fn, criterion, encoder_opt, head_opt
                )
                valid_metric = eval_regr(args, model, label_mean, label_std, valid_dataset, collate_fn, metric_name)
                test_metric = eval_regr(args, model, label_mean, label_std, test_dataset, collate_fn, metric_name)
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "valid": valid_metric, "test": test_metric})
            f.flush()
            print(
                f"epoch:{epoch} train/loss:{train_loss:.6g} "
                f"valid/{metric_name}:{valid_metric:.6g} test/{metric_name}:{test_metric:.6g}"
            )
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_train_epoch = epoch
                best_train_on_test = test_metric
            if is_better(valid_metric, best_valid, select_mode):
                best_valid = valid_metric
                best_valid_epoch = epoch
                best_valid_on_test = test_metric

    result = {
        "best_valid": best_valid,
        "best_valid_epoch": best_valid_epoch,
        "best_train_loss": best_train_loss,
        "best_train_epoch": best_train_epoch,
        "best_train_on_test": best_train_on_test,
        "best_valid_on_test": best_valid_on_test,
        "metric_name": metric_name,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=4), encoding="utf-8")
    print(f"Wrote {out_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
