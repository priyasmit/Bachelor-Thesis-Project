import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm import GNNExplainer

try:
    # Imports PyG explanation metrics
    from torch_geometric.explain.metric import (
        fidelity,
        characterization_score,
        fidelity_curve_auc,
        unfaithfulness,
    )
    HAS_EXPLAIN_METRICS = True
    EXPLAIN_METRICS_IMPORT_ERROR = None
except Exception as e:
    # If import fails, explanation metrics will be skipped
    HAS_EXPLAIN_METRICS = False
    EXPLAIN_METRICS_IMPORT_ERROR = str(e)

from dataset_pyg import ApkGraphDataset
from models_pyg import GCNGraphClassifier, GraphSAGEGraphClassifier



# General utilities
def set_seed(seed: int):
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_div(a, b):
    # Avoid division by zero
    return float(a / b) if b != 0 else 0.0


def tensor_to_float(x):
    # Converts the value to a Python float
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return None
        return float(x.detach().cpu().view(-1)[0].item())
    return float(x)


def threshold_mask(mask: Optional[torch.Tensor], threshold: float):
    # Turn a soft mask into a binary mask
    if mask is None:
        return None
    return (mask >= threshold).float()


def get_label_from_data(data) -> int:
    # Read graph label from a PyG Data object
    return int(data.y.view(-1)[0].item())


def mean_std_or_none(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    # Return mean and std if list is not empty
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def roc_auc_rank(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y_true = y_true.view(-1).int().cpu()
    y_score = y_score.view(-1).float().cpu()

    n = len(y_true)
    if n == 0:
        return 0.0

    n_pos = int((y_true == 1).sum().item())
    n_neg = int((y_true == 0).sum().item())

    # If only one class is present then AUC is not defined 
    if n_pos == 0 or n_neg == 0:
        return 0.0
        
    # Sort scores to assign ranks
    order = torch.argsort(y_score)
    sorted_scores = y_score[order]

    ranks = torch.zeros(n, dtype=torch.float32)
    i = 0
    cur_rank = 1
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j].item() == sorted_scores[i].item():
            j += 1

        # Average rank for tied scores
        avg_rank = (cur_rank + (cur_rank + (j - i) - 1)) / 2.0
        ranks[order[i:j]] = avg_rank

        cur_rank += (j - i)
        i = j

    # Compute AUC from positive ranks
    sum_ranks_pos = float(ranks[y_true == 1].sum().item())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def evaluate(model, loader, device):
    # Evaluate the model on a full dataset split
    model.eval()
    all_logits = []
    all_y = []
    all_pkg = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = batch.y.view(-1).long()

        all_logits.append(logits.cpu())
        all_y.append(y.cpu())
        all_pkg.extend(list(batch.pkg))

    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0)

    probs = torch.softmax(logits, dim=-1)
    probs_pos = probs[:, 1]
    y_pred = torch.argmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values

    # Confusion matrix 
    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())

    # Standard classification metrics
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    auc = roc_auc_rank(y_true, probs_pos)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_samples": int(len(y_true)),
    }

    # Save prediction details for each graph
    outputs = []
    for pkg, yt, yp, p, c in zip(
        all_pkg,
        y_true.tolist(),
        y_pred.tolist(),
        probs_pos.tolist(),
        confidence.tolist(),
    ):
        outputs.append({
            "pkg": pkg,
            "y_true": int(yt),
            "y_pred": int(yp),
            "p_malware": float(p),
            "confidence": float(c),
            "correct": bool(int(yt) == int(yp)),
        })

    return metrics, outputs



# Splitting
def stratified_split_indices(
    labels: List[int],
    seed: int = 123,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    # Split data while keeping class balance
    rng = random.Random(seed)

    idx_by_class = {}
    for i, y in enumerate(labels):
        idx_by_class.setdefault(int(y), []).append(i)

    train_idx, val_idx, test_idx = [], [], []

    for cls, idxs in idx_by_class.items():
        idxs = idxs[:]
        rng.shuffle(idxs)

        n = len(idxs)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def summarise_split(name: str, ds_part: List[Data]) -> Dict[str, int]:
    # Count samples per class in one split
    labels = [get_label_from_data(d) for d in ds_part]
    zeros = sum(1 for y in labels if y == 0)
    ones = sum(1 for y in labels if y == 1)
    return {"name": name, "n": len(ds_part), "class_0": zeros, "class_1": ones}


# Model wrapper for explainer
# Wrap graph classifier so PyG Explainer can call it with x, edge_index, batch.
class _ExplainWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch):
        # Rebuild a Data object for the model
        data = Data(x=x, edge_index=edge_index, batch=batch)
        return self.model(data)  # raw logits [num_graphs, 2]


def build_explainer(model, explain_epochs: int, explain_lr: float):
    # Build GNNExplainer with chosen settings
    wrapped = _ExplainWrapper(model)

    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(
            epochs=explain_epochs,
            lr=explain_lr,
        ),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
        # Keep raw masks during evaluation
        threshold_config=None,
    )
    return explainer



# Prediction collection for explanations
@torch.no_grad()
def collect_test_preds(model, dataset, device):
    # Run prediction on each test graph separately
    model.eval()
    rows = []

    for i, data in enumerate(dataset):
        data = data.to(device)
        batch = Batch.from_data_list([data]).to(device)

        logits = model(batch)
        probs = torch.softmax(logits, dim=-1)[0]
        prob_pos = float(probs[1].item())
        conf = float(probs.max().item())
        yp = int(torch.argmax(logits, dim=-1).item())
        yt = int(batch.y.view(-1).item())

        rows.append({
            "index": i,
            "pkg": str(getattr(data, "pkg", f"graph_{i}")),
            "y_true": yt,
            "y_pred": yp,
            "p_malware": prob_pos,
            "confidence": conf,
            "correct": bool(yt == yp),
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.size(1)),
        })

    return rows


def pick_graph_indices_for_explain(
    pred_rows: List[Dict],
    n: int,
    focus: str = "correct_balanced",
    confidence_min: float = 0.70,
    min_nodes: int = 5,
    min_edges: int = 4,
):
    # Select graphs that are suitable for explanation
    usable = []
    skipped = []

    for row in pred_rows:
        reason = None

        if row["num_nodes"] < min_nodes:
            reason = f"too_few_nodes<{min_nodes}"
        elif row["num_edges"] < min_edges:
            reason = f"too_few_edges<{min_edges}"
        elif row["confidence"] < confidence_min:
            reason = f"low_confidence<{confidence_min}"
        elif focus in {"correct", "correct_balanced"} and not row["correct"]:
            reason = "incorrect_prediction"

        if reason is None:
            usable.append(row)
        else:
            skipped.append({"index": row["index"], "pkg": row["pkg"], "reason": reason})

    if focus == "all":
        chosen_rows = usable[:n]

    elif focus == "correct":
        chosen_rows = usable[:n]

    elif focus == "errors":
        # Only explain wrong predictions
        err_rows = []
        for row in pred_rows:
            if row["num_nodes"] < min_nodes:
                continue
            if row["num_edges"] < min_edges:
                continue
            if row["confidence"] < confidence_min:
                continue
            if not row["correct"]:
                err_rows.append(row)
        chosen_rows = err_rows[:n]

    elif focus == "correct_balanced":
        # Try to choose a balanced number from each class
        cls0 = [r for r in usable if r["y_true"] == 0]
        cls1 = [r for r in usable if r["y_true"] == 1]

        cls0 = sorted(cls0, key=lambda r: r["confidence"], reverse=True)
        cls1 = sorted(cls1, key=lambda r: r["confidence"], reverse=True)

        half = math.ceil(n / 2)
        chosen_rows = cls0[:half] + cls1[:half]
        chosen_rows = chosen_rows[:n]

        # If one class has too few samples, fill remaining spots
        if len(chosen_rows) < n:
            chosen_idx = {r["index"] for r in chosen_rows}
            rest = [r for r in usable if r["index"] not in chosen_idx]
            rest = sorted(rest, key=lambda r: r["confidence"], reverse=True)
            need = n - len(chosen_rows)
            chosen_rows.extend(rest[:need])

    else:
        raise ValueError(f"Unknown explain focus: {focus}")

    chosen_indices = [r["index"] for r in chosen_rows]

    selection_summary = {
        "requested_n": int(n),
        "selected_n": int(len(chosen_indices)),
        "usable_n": int(len(usable)),
        "skipped_n": int(len(skipped)),
        "focus": focus,
        "confidence_min": float(confidence_min),
        "min_nodes": int(min_nodes),
        "min_edges": int(min_edges),
    }

    return chosen_indices, chosen_rows, skipped, selection_summary


def clone_with_edge_mask(explanation, edge_mask):
    # Copy explanation and replace edge mask
    new_exp = explanation.clone()
    new_exp.edge_mask = edge_mask
    return new_exp


def topk_binary_mask(mask: torch.Tensor, k: int) -> torch.Tensor:
    # Keep only the top-k highest mask values
    k = max(1, min(int(k), mask.numel()))
    topk_idx = torch.topk(mask, k).indices
    out = torch.zeros_like(mask)
    out[topk_idx] = 1.0
    return out



# Explanation metrics

def compute_explanation_metrics(
    explainer,
    explanation,
    curve_steps: int = 10,
    unfaith_topk: Optional[int] = None,
):
    # Compute several explanation quality metrics
    results = {
        "fidelity_pos": None,
        "fidelity_neg": None,
        "characterization_score": None,
        "fidelity_curve_auc": None,
        "unfaithfulness": None,
        "curve_x_frac_edges": None,
        "curve_fidelity_pos": None,
        "curve_fidelity_neg": None,
        "fidelity_error": None,
        "characterization_error": None,
        "fidelity_curve_auc_error": None,
        "unfaithfulness_error": None,
        "metric_import_error": None,
    }

    if not HAS_EXPLAIN_METRICS:
        results["metric_import_error"] = EXPLAIN_METRICS_IMPORT_ERROR
        return results

    # Base fidelity
    try:
        fid_pos, fid_neg = fidelity(explainer, explanation)
        results["fidelity_pos"] = tensor_to_float(fid_pos)
        results["fidelity_neg"] = tensor_to_float(fid_neg)
    except Exception as e:
        results["fidelity_error"] = str(e)

    # Characterization score
    try:
        if results["fidelity_pos"] is not None and results["fidelity_neg"] is not None:
            char = characterization_score(
                torch.tensor([results["fidelity_pos"]], dtype=torch.float),
                torch.tensor([results["fidelity_neg"]], dtype=torch.float),
            )
            results["characterization_score"] = tensor_to_float(char)
    except Exception as e:
        results["characterization_error"] = str(e)

    # Fidelity curve and AUC
    try:
        edge_mask = explanation.edge_mask
        if edge_mask is None:
            raise ValueError("No edge_mask available in explanation.")

        n_edges = int(edge_mask.numel())
        if n_edges < 2:
            raise ValueError("Too few edges for fidelity curve.")

        xs = torch.linspace(1.0 / curve_steps, 1.0, steps=curve_steps)
        curve_pos = []
        curve_neg = []

        for frac in xs:
            # Keep a bit of the most important edges
            k = max(1, int(round(float(frac.item()) * n_edges)))
            mask_k = topk_binary_mask(edge_mask, k)
            exp_k = clone_with_edge_mask(explanation, mask_k)

            p, n = fidelity(explainer, exp_k)
            curve_pos.append(tensor_to_float(p))
            curve_neg.append(tensor_to_float(n))

        results["curve_x_frac_edges"] = [float(v) for v in xs.tolist()]
        results["curve_fidelity_pos"] = [None if v is None else float(v) for v in curve_pos]
        results["curve_fidelity_neg"] = [None if v is None else float(v) for v in curve_neg]

        if all(v is not None for v in curve_pos) and all(v is not None for v in curve_neg):
            auc = fidelity_curve_auc(
                torch.tensor(curve_pos, dtype=torch.float),
                torch.tensor(curve_neg, dtype=torch.float),
                xs,
            )
            results["fidelity_curve_auc"] = tensor_to_float(auc)
        else:
            results["fidelity_curve_auc_error"] = "Curve contains invalid values."

    except Exception as e:
        results["fidelity_curve_auc_error"] = str(e)

    # Unfaithfulness
    try:
        unf = unfaithfulness(explainer, explanation, top_k=unfaith_topk)
        results["unfaithfulness"] = tensor_to_float(unf)
    except Exception as e:
        results["unfaithfulness_error"] = str(e)

    return results


def aggregate_explanation_metrics(rows):
    # Average explanation metrics over all explained graphs
    metric_keys = [
        "fidelity_pos",
        "fidelity_neg",
        "characterization_score",
        "fidelity_curve_auc",
        "unfaithfulness",
    ]
    error_keys = [
        "fidelity_error",
        "characterization_error",
        "fidelity_curve_auc_error",
        "unfaithfulness_error",
        "metric_import_error",
    ]

    out = {"num_explanations": len(rows)}

    for key in metric_keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        mean_v, std_v = mean_std_or_none(vals)
        out[f"mean_{key}"] = mean_v
        out[f"std_{key}"] = std_v
        out[f"num_valid_{key}"] = len(vals)

    for key in error_keys:
        count = sum(1 for r in rows if r.get(key) not in (None, ""))
        out[f"num_{key}"] = int(count)

    return out


def run_explanations(
    model,
    test_ds,
    device,
    out_dir: Path,
    explain_n: int,
    explain_epochs: int,
    explain_lr: float,
    explain_threshold: float,
    curve_steps: int,
    focus: str,
    confidence_min: float,
    min_nodes: int,
    min_edges: int,
    unfaith_topk=None,
):
    # Run GNNExplainer on selected test graphs
    model.eval()
    explainer = build_explainer(model, explain_epochs, explain_lr)

    pred_rows = collect_test_preds(model, test_ds, device)

    chosen_indices, chosen_rows, skipped_rows, selection_summary = pick_graph_indices_for_explain(
        pred_rows=pred_rows,
        n=explain_n,
        focus=focus,
        confidence_min=confidence_min,
        min_nodes=min_nodes,
        min_edges=min_edges,
    )

    exp_dir = out_dir / "explanations"
    exp_dir.mkdir(parents=True, exist_ok=True)

    per_graph_rows = []

    if len(chosen_indices) == 0:
        # Save empty summary if no graphs were suitable
        empty_summary = {
            "num_explanations": 0,
            "message": "No usable graphs selected for explanation. Lower confidence threshold or inspect test performance first.",
            **selection_summary,
        }

        with open(exp_dir / "selection_summary.json", "w", encoding="utf-8") as f:
            json.dump(selection_summary, f, indent=2)

        with open(exp_dir / "skipped_graphs.json", "w", encoding="utf-8") as f:
            json.dump(skipped_rows, f, indent=2)

        with open(exp_dir / "explanation_metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(empty_summary, f, indent=2)

        print("\n=== EXPLANATION METRICS ===")
        print("No usable graphs selected for explanation.")
        return

    for j, i in enumerate(chosen_indices, start=1):
        data = test_ds[i].to(device)
        batch_vec = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            batch=batch_vec,
        )

        metrics = compute_explanation_metrics(
            explainer=explainer,
            explanation=explanation,
            curve_steps=curve_steps,
            unfaith_topk=unfaith_topk,
        )

        # Save thresholded masks for easier later inspection
        node_mask_saved = threshold_mask(explanation.node_mask, explain_threshold)
        edge_mask_saved = threshold_mask(explanation.edge_mask, explain_threshold)

        pred_meta = chosen_rows[j - 1]
        pkg_str = pred_meta["pkg"]

        per_graph_row = {
            "pkg": pkg_str,
            "index": int(i),
            "y_true": int(pred_meta["y_true"]),
            "y_pred": int(pred_meta["y_pred"]),
            "p_malware": float(pred_meta["p_malware"]),
            "confidence": float(pred_meta["confidence"]),
            "correct": bool(pred_meta["correct"]),
            "num_nodes": int(pred_meta["num_nodes"]),
            "num_edges": int(pred_meta["num_edges"]),
            **metrics,
        }
        per_graph_rows.append(per_graph_row)

        # Save a text summary per graph
        txt_path = exp_dir / f"{j:03d}_{pkg_str}_metrics.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for k, v in per_graph_row.items():
                f.write(f"{k}: {v}\n")

        # Save masks and metadata as a PyTorch file
        save_obj = {
            "pkg": pkg_str,
            "index": int(i),
            "y_true": int(pred_meta["y_true"]),
            "y_pred": int(pred_meta["y_pred"]),
            "p_malware": float(pred_meta["p_malware"]),
            "confidence": float(pred_meta["confidence"]),
            "correct": bool(pred_meta["correct"]),
            "metrics": metrics,
            "node_mask_thresholded": None if node_mask_saved is None else node_mask_saved.detach().cpu(),
            "edge_mask_thresholded": None if edge_mask_saved is None else edge_mask_saved.detach().cpu(),
            "edge_index": data.edge_index.detach().cpu(),
        }
        torch.save(save_obj, exp_dir / f"{j:03d}_{pkg_str}.pt")

    summary = aggregate_explanation_metrics(per_graph_rows)
    summary.update(selection_summary)

    print("\n=== EXPLANATION METRICS ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    with open(exp_dir / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(selection_summary, f, indent=2)

    with open(exp_dir / "skipped_graphs.json", "w", encoding="utf-8") as f:
        json.dump(skipped_rows, f, indent=2)

    with open(exp_dir / "explanation_metrics_per_graph.json", "w", encoding="utf-8") as f:
        json.dump(per_graph_rows, f, indent=2)

    with open(exp_dir / "explanation_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)



# Training

def build_model(model_name: str, in_dim: int, hidden_dim: int, dropout: float):
    # Build the requested model
    if model_name == "gcn":
        return GCNGraphClassifier(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    if model_name == "sage":
        return GraphSAGEGraphClassifier(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unknown model: {model_name}")


def choose_checkpoint_score(val_metrics: Dict, monitor: str) -> float:
    # Use validation ROC-AUC to choose the best model
    return float(val_metrics["roc_auc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory with *_graph.pkl, *_features.npy, *_label.json")
    ap.add_argument("--model", choices=["gcn", "sage"], default="gcn")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dir", default="runs_pyg")

    # Early stopping settings
    ap.add_argument("--monitor", choices=["auc"], default="auc")
    ap.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    ap.add_argument("--min_delta", type=float, default=1e-4, help="Minimum improvement to reset patience")

    # Explanation settings
    ap.add_argument("--explain", action="store_true", help="Run explanation analysis on selected test graphs")
    ap.add_argument("--explain_n", type=int, default=12, help="How many test graphs to explain")
    ap.add_argument("--explain_epochs", type=int, default=200, help="GNNExplainer epochs per graph")
    ap.add_argument("--explain_focus", choices=["correct_balanced", "correct", "errors", "all"], default="correct_balanced")
    ap.add_argument("--explain_lr", type=float, default=0.01)
    ap.add_argument("--explain_threshold", type=float, default=0.5)
    ap.add_argument("--explain_curve_steps", type=int, default=10)
    ap.add_argument("--explain_unfaith_topk", type=int, default=None)
    ap.add_argument("--explain_confidence_min", type=float, default=0.70)
    ap.add_argument("--explain_min_nodes", type=int, default=5)
    ap.add_argument("--explain_min_edges", type=int, default=4)

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset and make train/val/test split
    ds = ApkGraphDataset(args.data_dir)
    labels = [get_label_from_data(ds[i]) for i in range(len(ds))]
    train_idx, val_idx, test_idx = stratified_split_indices(labels, seed=args.seed)

    train_ds = [ds[i] for i in train_idx]
    val_ds = [ds[i] for i in val_idx]
    test_ds = [ds[i] for i in test_idx]

    split_info = {
        "train": summarise_split("train", train_ds),
        "val": summarise_split("val", val_ds),
        "test": summarise_split("test", test_ds),
    }

    print("\n=== DATA SPLIT ===")
    for name in ["train", "val", "test"]:
        info = split_info[name]
        print(f"{name}: n={info['n']} class_0={info['class_0']} class_1={info['class_1']}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    in_dim = train_ds[0].x.size(-1)
    model = build_model(args.model, in_dim=in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    # Adam optimizer with weight decay for regularization
    opt = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=args.weight_decay,
    )   
    out_dir = Path(args.out_dir) / f"{args.model}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pt"

    # Save split information
    with open(out_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    best_score = -float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_graphs = 0

        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y.view(-1).long()
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * int(y.size(0))
            total_graphs += int(y.size(0))

        train_loss = safe_div(total_loss, total_graphs)

        # Validate after each epoch
        val_metrics, _ = evaluate(model, val_loader, device)
        score = choose_checkpoint_score(val_metrics, args.monitor)

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "checkpoint_score": score,
        }
        history.append(epoch_row)

        # Save model if validation score improves
        improved = (score - best_score) > args.min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "monitor": args.monitor,
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f} | "
            f"score={score:.4f}"
        )

        # Stop early if validation score has not improved
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (best epoch: {best_epoch}, monitor: {args.monitor})")
            break

    # Save full training history
    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Load best saved model before test evaluation
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    print(f"\nLoaded best model from epoch {ckpt.get('best_epoch', 'unknown')} using monitor={ckpt.get('monitor', args.monitor)}")

    test_metrics, test_outputs = evaluate(model, test_loader, device)

    print("\n=== TEST METRICS ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Save final test results
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    with open(out_dir / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(test_outputs, f, indent=2)

    print(f"\nSaved to: {out_dir}")

    # Optionally run explanation analysis
    if args.explain:
        run_explanations(
            model=model,
            test_ds=test_ds,
            device=device,
            out_dir=out_dir,
            explain_n=args.explain_n,
            explain_epochs=args.explain_epochs,
            explain_lr=args.explain_lr,
            explain_threshold=args.explain_threshold,
            curve_steps=args.explain_curve_steps,
            focus=args.explain_focus,
            confidence_min=args.explain_confidence_min,
            min_nodes=args.explain_min_nodes,
            min_edges=args.explain_min_edges,
            unfaith_topk=args.explain_unfaith_topk,
        )


if __name__ == "__main__":
    main()
