
# run example:
# python view_explanations.py --exp_dir runs_pyg/sage_seed123/explanations --topk 10
# python view_explanations.py --exp_dir runs_pyg/gcn_seed123/explanations --topk 10

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:
    torch = None


def to_numpy(x: Any):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return None


def load_file(path: Path):
    suffix = path.suffix.lower()

    if suffix == ".json":
        return json.load(open(path))

    if suffix in (".pkl", ".pickle"):
        return pickle.load(open(path, "rb"))

    if suffix in (".pt", ".pth"):
        return torch.load(path, map_location="cpu")

    raise ValueError("Unsupported file type")


def topk(scores, k):
    scores = np.asarray(scores).reshape(-1)
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


def inspect_file(path: Path, topk_n: int):

    obj = load_file(path)

    if not isinstance(obj, dict):
        obj = vars(obj)

    edge_index = to_numpy(obj.get("edge_index"))
    edge_mask = to_numpy(obj.get("edge_mask"))
    node_mask = to_numpy(obj.get("node_mask"))

    print("=" * 80)
    print(f"File: {path}")
    print("-" * 80)

    print("Detected keys:", list(obj.keys()))

    if node_mask is not None:

        node_mask = np.asarray(node_mask)

        if node_mask.ndim == 2:
            node_mask = node_mask.mean(axis=1)

        print("\nNode mask shape:", node_mask.shape)

        idx, vals = topk(node_mask, topk_n)

        print(f"Top-{len(idx)} important nodes (index -> score):")
        for i, v in zip(idx, vals):
            print(f"{int(i)} -> {v:.6f}")

    if edge_mask is not None:

        edge_mask = np.asarray(edge_mask)

        print("\nEdge mask shape:", edge_mask.shape)

        idx, vals = topk(edge_mask, topk_n)

        print(f"Top-{len(idx)} important edges (src -> dst -> score):")

        if edge_index is not None and edge_index.shape[0] == 2:

            for e_i, v in zip(idx, vals):
                src = int(edge_index[0, e_i])
                dst = int(edge_index[1, e_i])
                print(f"{src} -> {dst} score={v:.6f}")

        else:
            for e_i, v in zip(idx, vals):
                print(f"{int(e_i)} -> {v:.6f}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--file", default="")

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)

    files = sorted(
        p for p in exp_dir.iterdir()
        if p.suffix.lower() in [".pt", ".pth", ".pkl", ".pickle", ".json"]
    )

    if args.file:
        files = [p for p in files if p.name.startswith(args.file)]

    for f in files:
        inspect_file(f, args.topk)


if __name__ == "__main__":
    main()