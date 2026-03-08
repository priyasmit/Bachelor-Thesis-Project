# run example: python build_gnn_dataset.py --config path/to/config.json --out_dir path/to/gnn_preprocessed --dim 32

# Inputs:
# - config.json: list of graphs with graph_path, embedding_path, label
# - graph_path: .gml call graph
# - embedding_path: node2vec word2vec-style .vec text file

# Outputs (in out_dir):
# - <pkg>_graph.pkl      : pickled NetworkX graph
# - <pkg>_features.npy   : [num_nodes, dim] float32
# - <pkg>_label.json     : {"label": 0/1}
#

from pathlib import Path
import json
import pickle
import re
import numpy as np
import networkx as nx

#Cleans node ids so they match between GML and node2vec vec file.
def clean_node_id(raw: str) -> str:
    
    s = str(raw)
    s = re.sub(r"\[.*?\]", "", s)                 # remove [..]
    s = re.sub(r"@0x[0-9a-fA-F]+", "", s)         # remove @0x...
    return s.strip()

    # Reads word2vec-style node2vec file:
    #   first line: "<num_nodes> <dim>"
    #   then lines: "<node_id> v1 v2 ... v_dim"
    # Returns dict: cleaned_node_id -> np.array([dim], float32)
def read_node2vec_vec(vec_path: Path, dim: int) -> dict:
    emb = {}
    with open(vec_path, "r") as f:
        _header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < dim + 1:
                continue
            node_raw = parts[0]
            node = clean_node_id(node_raw)
            try:
                vec = np.asarray(list(map(float, parts[-dim:])), dtype=np.float32)
            except ValueError:
                continue
            emb[node] = vec
    return emb

# Make a stable package/app id from the filename.
def derive_pkg_name(gml_path: Path) -> str:
    stem = gml_path.stem
    stem = stem.replace(".apk_CG", "").replace("_CG", "")
    return stem


def main(config_path: str, out_dir: str, dim: int = 32):
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        config = json.load(f)

    graphs = config.get("graphs", [])
    if not graphs:
        raise ValueError(f"No 'graphs' list found in {config_path}")

    manifest = []

    for item in graphs:
        gml_path = Path(item["graph_path"])
        vec_path = Path(item["embedding_path"])
        label = int(item["label"])

        if not gml_path.exists():
            print(f"[SKIP] missing graph: {gml_path}")
            continue
        if not vec_path.exists():
            print(f"[SKIP] missing embeddings: {vec_path}")
            continue

        pkg = derive_pkg_name(gml_path)

        graph_out = out_dir / f"{pkg}_graph.pkl"
        feat_out = out_dir / f"{pkg}_features.npy"
        label_out = out_dir / f"{pkg}_label.json"

        # ---- skip if already built ----
        if graph_out.exists() and feat_out.exists() and label_out.exists():
            print(f"[SKIP] already exists: {pkg}")
            manifest.append({
                "pkg": pkg,
                "graph_path": str(graph_out),
                "features_path": str(feat_out),
                "label_path": str(label_out),
                "label": label,
                "skipped": True
            })
            continue

        # ---- Build outputs ----
        G = nx.read_gml(str(gml_path))
        emb = read_node2vec_vec(vec_path, dim=dim)

        nodes = list(G.nodes())
        X = np.zeros((len(nodes), dim), dtype=np.float32)

        missing = 0
        for i, n in enumerate(nodes):
            key = clean_node_id(n)
            v = emb.get(key)

            if v is None:
                # fallback: try node attribute
                try:
                    key2 = clean_node_id(G.nodes[n].get("orig_id", n))
                    v = emb.get(key2)
                except Exception:
                    v = None

            if v is None:
                missing += 1
                continue
            X[i] = v

        with open(graph_out, "wb") as f:
            pickle.dump(G, f)
        np.save(feat_out, X)
        with open(label_out, "w") as f:
            json.dump({"label": label}, f)

        print(f"[OK] {pkg}: nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
              f"missing_emb={missing}/{len(nodes)} label={label}")

        manifest.append({
            "pkg": pkg,
            "graph_path": str(graph_out),
            "features_path": str(feat_out),
            "label_path": str(label_out),
            "label": label,
            "num_nodes": int(G.number_of_nodes()),
            "num_edges": int(G.number_of_edges()),
            "missing_emb": int(missing),
            "skipped": False
        })

    # Write manifest once after processing all graphs
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Wrote/verified {len(manifest)} entries in: {out_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.json with graphs list")
    ap.add_argument("--out_dir", required=True, help="Output directory (e.g., .../gcn_tmp)")
    ap.add_argument("--dim", type=int, default=32, help="Node2vec embedding dimension")
    args = ap.parse_args()

    main(args.config, args.out_dir, dim=args.dim)