
# Run examples:
# python train_eval_pyg.py --model gcn --data_dir /path/to/gcn_tmp --epochs 50
# python train_eval_pyg.py --model sage --data_dir /path/to/gcn_tmp --epochs 50

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm import GNNExplainer

from dataset_pyg import ApkGraphDataset
from models_pyg import GCNGraphClassifier, GraphSAGEGraphClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_div(a, b):
    return float(a / b) if b != 0 else 0.0


def roc_auc_rank(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y_true = y_true.view(-1).int()
    y_score = y_score.view(-1).float()
    n_pos = int((y_true == 1).sum().item())
    n_neg = int((y_true == 0).sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    sorted_idx = torch.argsort(y_score)
    ranks = torch.empty_like(sorted_idx, dtype=torch.float)
    ranks[sorted_idx] = torch.arange(1, len(y_score) + 1, dtype=torch.float)

    sum_ranks_pos = float(ranks[y_true == 1].sum().item())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_y = []
    all_pkg = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch) # [B, 2]
        y = batch.y.view(-1).long() # [B]
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())
        all_pkg.extend(list(batch.pkg))

    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0)

    probs_pos = torch.softmax(logits, dim=-1)[:, 1]
    y_pred = torch.argmax(logits, dim=-1)

    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())

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
    }

    outputs = []
    for pkg, yt, yp, p in zip(all_pkg, y_true.tolist(), y_pred.tolist(), probs_pos.tolist()):
        outputs.append({
            "pkg": pkg,
            "y_true": int(yt),
            "y_pred": int(yp),
            "p_malware": float(p),
        })

    return metrics, outputs


def split_indices(n, seed=123, train_ratio=0.7, val_ratio=0.15):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


# Wraps the graph-level classifier so PyG Explainer can call it
class _ExplainWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch):
        data = Data(x=x, edge_index=edge_index, batch=batch)
        return self.model(data)  # raw logits [num_graphs, 2]


def build_explainer(model, explain_epochs: int):
    wrapped = _ExplainWrapper(model)

    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=explain_epochs),
        explanation_type="model",
        node_mask_type="attributes",  # feature importance per node feature
        edge_mask_type="object", # importance per edge
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw", # model returns logits
        ),
    )
    return explainer


@torch.no_grad()
# Gets y_true, y_pred, probs_pos for each graph in a safe single-graph batch pass.
def collect_test_preds(model, dataset, device):

    model.eval()
    ys, yps, ps = [], [], []
    for data in dataset:
        data = data.to(device)
        batch = Batch.from_data_list([data]).to(device)

        logits = model(batch)  # [1, 2]
        prob_pos = torch.softmax(logits, dim=-1)[0, 1].item()
        yp = int(torch.argmax(logits, dim=-1).item())
        yt = int(batch.y.view(-1).item())

        ys.append(yt)
        yps.append(yp)
        ps.append(prob_pos)
    return ys, yps, ps


def pick_graph_indices_for_explain(test_ds, y_true_list, y_pred_list, n, focus="errors"):
    idx_all = list(range(len(test_ds)))

    if focus == "all":
        return idx_all[:n]

    # focus == "errors": prioritize FN/FP, then fill with TP/TN
    fp = [i for i in idx_all if (y_true_list[i] == 0 and y_pred_list[i] == 1)]
    fn = [i for i in idx_all if (y_true_list[i] == 1 and y_pred_list[i] == 0)]
    tp = [i for i in idx_all if (y_true_list[i] == 1 and y_pred_list[i] == 1)]
    tn = [i for i in idx_all if (y_true_list[i] == 0 and y_pred_list[i] == 0)]

    picked = (fn + fp + tp + tn)[:n]
    return picked


def run_explanations(model, test_ds, device, out_dir: Path, explain_n: int, explain_epochs: int, focus: str):
    model.eval()
    explainer = build_explainer(model, explain_epochs=explain_epochs)

    # Collect predictions so we can prioritize mistakes
    y_true_list, y_pred_list, p_list = collect_test_preds(model, test_ds, device)
    chosen = pick_graph_indices_for_explain(test_ds, y_true_list, y_pred_list, explain_n, focus=focus)

    exp_dir = out_dir / "explanations"
    exp_dir.mkdir(parents=True, exist_ok=True)

    for j, i in enumerate(chosen, start=1):
        data = test_ds[i].to(device)

        # Explainer expects x, edge_index, batch
        batch_vec = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

        # Explain the model's predicted class
        target = int(y_pred_list[i])

        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            batch=batch_vec,
            target=target,
        )

        pkg = getattr(data, "pkg", f"graph_{i}")
        pkg_str = pkg if isinstance(pkg, str) else str(pkg)

        save_obj = {
            "pkg": pkg_str,
            "index": i,
            "y_true": int(y_true_list[i]),
            "y_pred": int(y_pred_list[i]),
            "p_malware": float(p_list[i]),
            "target_explained": int(target),
            "node_mask": None if explanation.node_mask is None else explanation.node_mask.detach().cpu(),
            "edge_mask": None if explanation.edge_mask is None else explanation.edge_mask.detach().cpu(),
            "edge_index": data.edge_index.detach().cpu(),
            "num_nodes": int(data.num_nodes),
        }

        torch.save(save_obj, exp_dir / f"{j:03d}_{pkg_str}.pt")

        # Optional readable summary
        txt_path = exp_dir / f"{j:03d}_{pkg_str}_summary.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"pkg: {pkg_str}\n")
            f.write(f"index: {i}\n")
            f.write(f"y_true: {int(y_true_list[i])}\n")
            f.write(f"y_pred: {int(y_pred_list[i])}\n")
            f.write(f"p_malware: {float(p_list[i]):.6f}\n")
            f.write(f"target_explained: {int(target)}\n\n")

            if explanation.node_mask is not None:
                node_mask = explanation.node_mask.detach().cpu()

                if node_mask.dim() == 2:
                    # node_mask_type="attributes": [num_nodes, num_features]
                    node_scores = node_mask.mean(dim=1)
                else:
                    node_scores = node_mask.view(-1)

                topn = min(10, node_scores.numel())
                vals, idxs = torch.topk(node_scores, k=topn)

                f.write("Top nodes:\n")
                for idx_, val_ in zip(idxs.tolist(), vals.tolist()):
                    f.write(f"  node {idx_}: {val_:.6f}\n")

            if explanation.edge_mask is not None:
                edge_mask = explanation.edge_mask.detach().cpu().view(-1)
                edge_index_cpu = data.edge_index.detach().cpu()

                tope = min(10, edge_mask.numel())
                vals, idxs = torch.topk(edge_mask, k=tope)

                f.write("\nTop edges:\n")
                for e_idx, val_ in zip(idxs.tolist(), vals.tolist()):
                    src = int(edge_index_cpu[0, e_idx])
                    dst = int(edge_index_cpu[1, e_idx])
                    f.write(f"  {src} -> {dst}: {val_:.6f}\n")

    print(f"Saved explanations to: {exp_dir} (count={len(chosen)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory with *_graph.pkl, *_features.npy, *_label.json")
    ap.add_argument("--model", choices=["gcn", "sage"], default="gcn")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dir", default="runs_pyg")

    ap.add_argument("--explain", action="store_true", help="Run torch_geometric.explain on test graphs")
    ap.add_argument("--explain_n", type=int, default=10, help="How many test graphs to explain")
    ap.add_argument("--explain_epochs", type=int, default=100, help="GNNExplainer epochs per graph")
    ap.add_argument(
        "--explain_focus",
        choices=["errors", "all"],
        default="errors",
        help="Explain errors first (FP/FN) or just first N graphs",
    )

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ApkGraphDataset(args.data_dir)
    train_idx, val_idx, test_idx = split_indices(len(ds), seed=args.seed)

    train_ds = [ds[i] for i in train_idx]
    val_ds = [ds[i] for i in val_idx]
    test_ds = [ds[i] for i in test_idx]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    in_dim = train_ds[0].x.size(-1)

    if args.model == "gcn":
        model = GCNGraphClassifier(in_dim=in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    else:
        model = GraphSAGEGraphClassifier(in_dim=in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    out_dir = Path(args.out_dir) / f"{args.model}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y.view(-1).long()
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        val_metrics, _ = evaluate(model, val_loader, device)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({"state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)

        print(
            f"Epoch {epoch:03d} | loss={total_loss:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['roc_auc']:.4f}"
        )

    # Load best checkpoint and test
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    test_metrics, test_outputs = evaluate(model, test_loader, device)

    print("\n=== TEST METRICS ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    with open(out_dir / "test_predictions.json", "w") as f:
        json.dump(test_outputs, f, indent=2)

    print(f"\nSaved to: {out_dir}")

    if args.explain:
        run_explanations(
            model=model,
            test_ds=test_ds,
            device=device,
            out_dir=out_dir,
            explain_n=args.explain_n,
            explain_epochs=args.explain_epochs,
            focus=args.explain_focus,
        )


if __name__ == "__main__":
    main()