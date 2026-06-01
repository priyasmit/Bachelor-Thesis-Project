from pathlib import Path
import json
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


class ApkGraphDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.graph_files = sorted(self.root_dir.glob("*_graph.pkl"))
        if not self.graph_files:
            raise FileNotFoundError(f"No *_graph.pkl files found in {self.root_dir}")

        # Derive package names from filenames
        self.pkgs = [gf.stem.replace("_graph", "") for gf in self.graph_files]

    def __len__(self):
        return len(self.pkgs)

    def __getitem__(self, idx: int) -> Data:
        pkg = self.pkgs[idx]

        graph_path = self.root_dir / f"{pkg}_graph.pkl"
        feats_path = self.root_dir / f"{pkg}_features.npy"
        label_path = self.root_dir / f"{pkg}_label.json"

        # Load graph (NetworkX)
        G = pickle.load(open(graph_path, "rb"))

        # Load features [N, F]
        x_np = np.load(feats_path)
        if x_np.ndim != 2:
            raise ValueError(f"{feats_path} must be 2D [num_nodes, feat_dim], got shape {x_np.shape}")

        # Load label
        label = int(json.load(open(label_path, "r"))["label"])
        y = torch.tensor([label], dtype=torch.long)  # graph-level label

        # Build edge_index [2, E]
        nodes = list(G.nodes())
        node_to_i = {n: i for i, n in enumerate(nodes)}
        edges = []
        for u, v in G.edges():
            edges.append((node_to_i[u], node_to_i[v]))
            edges.append((node_to_i[v], node_to_i[u]))  # undirected -> add both

        if len(edges) == 0:
            # Handle graphs with no edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        x = torch.tensor(x_np, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.pkg = pkg  # keep identifier for later reporting
        return data
