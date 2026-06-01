
import json
from pathlib import Path
import pandas as pd

graph_dir = Path("path/to/apks_cg") #CUSTOMIZE PATHS
emb_dir = Path("path/to/embeddings")
model_dir = Path("path/to/models")
edge_dir = Path("path/to/edges")
apk_csv = Path("path/to/apk.csv")
tmp_gcn = Path("path/to/gnn_preprocessed")
out_json = Path("path/to/config.json")
gml_files = list(graph_dir.glob("*.gml"))
print("Number of GML-files:", len(gml_files))
print("Example GML-files:", [f.name for f in gml_files[:5]])

# read CSV
df = pd.read_csv(apk_csv)
df["pkg_name_clean"] = df["pkg_name"].astype(str).str.replace('"', '').str.strip()
# label: 1 if vt_detection > 0, else 0
df["label"] = (df["vt_detection"] > 0).astype(int)

# Make dict: pkg_name_clean -> label
df["sha256_clean"] = df["sha256"].astype(str).str.strip()
csv_label_map = dict(zip(df["sha256_clean"], df["label"]))
print("example CSV sha256:", df["sha256_clean"].head(5).tolist())

tmp = []
for g in gml_files[:5]:
    name = g.stem
    if name.endswith("_CG"):
        name = name[:-3]
    if name.endswith(".apk"):
        name = name[:-4]
    tmp.append(name)
print("Example GML keys:", tmp)

# Determine pkg_names in current subset 
gml_files = sorted(graph_dir.glob("*.gml"))
gml_pkg_names = set()
for gml_file in gml_files:
    name = gml_file.stem
    if name.endswith("_CG"):
        name = name[:-3]  # strip "_CG"
    if name.endswith(".apk"):
        name = name[:-4]  # strip ".apk"
    gml_pkg_names.add(name)

# Filter CSV on only de pkg_names we have GML files for
subset_label_map = {k: v for k, v in csv_label_map.items() if k in gml_pkg_names}

configs = []
for gml_file in gml_files:
    name = gml_file.stem
    if name.endswith("_CG"):
        name = name[:-3]
    if name.endswith(".apk"):
        name = name[:-4]

    label = subset_label_map.get(name)
    if label is None:
        print(f"Geen label gevonden voor {name}, overslaan")
        continue

    configs.append({
        "graph_path": str(gml_file),
        "embedding_path": str(emb_dir / f"{name}.txt.vec"),
        "model_path": str(model_dir / f"{name}.model"),
        "edge_embedding_path": str(edge_dir / f"{name}_edges.txt"),
        "label": int(label)
    })

# write config.json
with out_json.open("w") as f:
    json.dump({"graphs": configs}, f, indent=2)

print(f" Config.json created with {len(configs)} graphs")
