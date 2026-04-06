import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tordata
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent

import sys

sys.path.append(str(REPO_ROOT / "demo" / "libs"))
sys.path.append(str(REPO_ROOT))

from datasets.pretreatment import pretreat
from opengait.utils import config_loader
from demo.libs.model import baselineDemo
from demo.libs.model.dataset import DataSet


def validate_raw_layout(raw_root: Path) -> None:
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw CASIA-B root does not exist: {raw_root}")

    sample = next(raw_root.rglob("*.png"), None)
    if sample is None:
        raise FileNotFoundError(f"No PNG files were found under {raw_root}")

    rel = sample.relative_to(raw_root)
    parts = rel.parts
    if len(parts) < 4:
        raise ValueError(
            "CASIA-B layout does not match subject/type/view/frame.png. "
            f"Example file: {sample}"
        )

    print("Raw CASIA-B root:", raw_root)
    print("Sample sequence path:", rel)


def ensure_pkl_dataset(raw_root: Path, pkl_root: Path, workers: int) -> None:
    sample_pkl = next(pkl_root.rglob("*.pkl"), None) if pkl_root.exists() else None
    if sample_pkl is not None:
        print("Found existing preprocessed dataset:", pkl_root)
        return

    print("Preprocessing CASIA-B into pickle format...")
    pretreat(raw_root, pkl_root, img_size=64, workers=workers, dataset="CASIAB")
    sample_pkl = next(pkl_root.rglob("*.pkl"), None)
    if sample_pkl is None:
        raise RuntimeError(f"Pretreatment completed but no .pkl files were created in {pkl_root}")


def build_cfg(dataset_root: Path, dataset_partition: Path) -> dict:
    cfgs = config_loader(str(REPO_ROOT / "configs" / "gaitbase" / "gaitbase_da_gait3d.yaml"))
    cfgs["data_cfg"]["dataset_name"] = "CASIA-B"
    cfgs["data_cfg"]["test_dataset_name"] = "CASIA-B"
    cfgs["data_cfg"]["dataset_root"] = str(dataset_root)
    cfgs["data_cfg"]["dataset_partition"] = str(dataset_partition)
    cfgs["data_cfg"]["num_workers"] = 1
    cfgs["data_cfg"]["cache"] = False
    cfgs["evaluator_cfg"]["enable_float16"] = torch.cuda.is_available()
    cfgs["evaluator_cfg"]["restore_hint"] = "load_current_gaitbase"
    cfgs["evaluator_cfg"]["save_name"] = "GaitBase_DA"
    cfgs["evaluator_cfg"]["metric"] = "euc"
    cfgs["evaluator_cfg"]["transform"] = [{"type": "BaseSilTransform"}]
    return cfgs


def build_loader(cfgs: dict, batch_size: int):
    dataset = DataSet(cfgs["data_cfg"])
    sampler_cfg = dict(cfgs["evaluator_cfg"]["sampler"])
    sampler_cfg["batch_size"] = batch_size
    loader = tordata.DataLoader(
        dataset=dataset,
        sampler=tordata.SequentialSampler(dataset),
        collate_fn=baselineDemo.CollateFn(dataset.label_set, sampler_cfg),
        num_workers=cfgs["data_cfg"]["num_workers"],
        batch_size=batch_size,
    )
    return loader


def extract_features(model, loader):
    embeddings = []
    labels = []
    types = []
    views = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting CASIA-B features"):
            inputs = model.inputs_pretreament(batch)
            retval, _ = model.forward(inputs)
            batch_embeddings = retval["inference_feat"]["embeddings"].detach().cpu().numpy()
            batch_labels = batch[1]
            batch_types = batch[2]
            batch_views = batch[3]
            for emb, lab, typ, vie in zip(batch_embeddings, batch_labels, batch_types, batch_views):
                embeddings.append(emb)
                labels.append(lab)
                types.append(typ)
                views.append(vie)

    return {
        "embeddings": np.asarray(embeddings),
        "labels": np.asarray(labels),
        "types": np.asarray(types),
        "views": np.asarray(views),
    }


def pairwise_distance(x: np.ndarray, y: np.ndarray, metric: str = "euc") -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tx = torch.from_numpy(x).to(device)
    ty = torch.from_numpy(y).to(device)
    if metric == "cos":
        tx = torch.nn.functional.normalize(tx, p=2, dim=1)
        ty = torch.nn.functional.normalize(ty, p=2, dim=1)
    num_bin = tx.size(2)
    dist = torch.zeros(tx.size(0), ty.size(0), device=device)
    for i in range(num_bin):
        _x = tx[:, :, i]
        _y = ty[:, :, i]
        if metric == "cos":
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            cur = (
                torch.sum(_x ** 2, 1).unsqueeze(1)
                + torch.sum(_y ** 2, 1).unsqueeze(0)
                - 2 * torch.matmul(_x, _y.transpose(0, 1))
            )
            dist += torch.sqrt(torch.relu(cur))
    return 1 - dist / num_bin if metric == "cos" else dist / num_bin


def de_diag(acc: np.ndarray, each_angle: bool = False):
    dividend = acc.shape[1] - 1.0
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result


def evaluate_casiab(features: dict, metric: str = "euc") -> dict:
    probe_seq_dict = {
        "NM": ["nm-05", "nm-06"],
        "BG": ["bg-01", "bg-02"],
        "CL": ["cl-01", "cl-02"],
    }
    gallery_seq = ["nm-01", "nm-02", "nm-03", "nm-04"]

    labels = features["labels"]
    seq_type = features["types"]
    view = features["views"]
    view_list = sorted(np.unique(view))
    acc = {}

    for type_name, probe_seq in probe_seq_dict.items():
        acc[type_name] = np.zeros((len(view_list), len(view_list))) - 1.0
        for v1, probe_view in enumerate(view_list):
            probe_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
            probe_x = features["embeddings"][probe_mask]
            probe_y = labels[probe_mask]

            for v2, gallery_view in enumerate(view_list):
                gallery_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                gallery_x = features["embeddings"][gallery_mask]
                gallery_y = labels[gallery_mask]
                if probe_x.size == 0 or gallery_x.size == 0:
                    continue
                dist = pairwise_distance(probe_x, gallery_x, metric)
                idx = dist.sort(1)[1].cpu().numpy()
                acc[type_name][v1, v2] = np.round(
                    np.sum(
                        np.cumsum(
                            np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:1]],
                            axis=1,
                        )
                        > 0,
                        axis=0,
                    )
                    * 100
                    / dist.shape[0],
                    2,
                )

    results = {}
    print("=== CASIA-B Rank-1 (Exclude identical-view cases) ===")
    for type_name in ("NM", "BG", "CL"):
        mean_acc = float(de_diag(acc[type_name]))
        angle_acc = de_diag(acc[type_name], each_angle=True)
        results[type_name] = mean_acc
        print(f"{type_name}: {mean_acc:.2f}%")
        print(f"{type_name} by view: {np.array2string(angle_acc, precision=2)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate the current demo GaitBase checkpoint on CASIA-B.")
    parser.add_argument(
        "--raw-root",
        default=str(PROJECT_ROOT / "casiab"),
        help="Path to the raw CASIA-B silhouette PNG dataset.",
    )
    parser.add_argument(
        "--pkl-root",
        default=str(PROJECT_ROOT / "casiab-pkl"),
        help="Path to the preprocessed CASIA-B pickle dataset.",
    )
    parser.add_argument(
        "--partition",
        default=str(REPO_ROOT / "datasets" / "CASIA-B" / "CASIA-B.json"),
        help="Path to the CASIA-B train/test split JSON.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of pretreatment workers.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size for CASIA-B evaluation.")
    parser.add_argument(
        "--skip-pretreatment",
        action="store_true",
        help="Skip PNG-to-PKL pretreatment and use an existing pkl dataset.",
    )
    parser.add_argument(
        "--save-json",
        default=str(REPO_ROOT / "output" / "casiab_gaitbase_eval.json"),
        help="Where to save the evaluation summary JSON.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    pkl_root = Path(args.pkl_root).resolve()
    partition = Path(args.partition).resolve()
    save_json = Path(args.save_json).resolve()

    os.chdir(REPO_ROOT)

    validate_raw_layout(raw_root)
    if not args.skip_pretreatment:
        ensure_pkl_dataset(raw_root, pkl_root, args.workers)

    cfgs = build_cfg(pkl_root, partition)
    model = baselineDemo.BaselineDemo(cfgs, training=False)
    model.requires_grad_(False)
    model.eval()

    loader = build_loader(cfgs, args.batch_size)
    features = extract_features(model, loader)
    results = evaluate_casiab(features, cfgs["evaluator_cfg"]["metric"])

    save_json.parent.mkdir(parents=True, exist_ok=True)
    with open(save_json, "w") as f:
        json.dump(
            {
                "raw_root": str(raw_root),
                "pkl_root": str(pkl_root),
                "partition": str(partition),
                "checkpoint": str((REPO_ROOT / "demo" / "checkpoints" / "gait_model" / "GaitBase_DA-180000.pt").resolve()),
                "results": results,
            },
            f,
            indent=2,
        )
    print("Saved evaluation summary to:", save_json)


if __name__ == "__main__":
    main()
