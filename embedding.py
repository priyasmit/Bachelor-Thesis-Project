
import json
from pathlib import Path
import networkx as nx
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from gensim.models import Word2Vec
from tqdm import tqdm

DIM = 32
WALK_LENGTH = 20
NUM_WALKS = 50
WINDOW = 10
BATCH_WORDS = 4
EDGE_BATCH_SIZE = 50000

CONFIG_PATH = Path("path/to/config.json") #CUSTOMISE PATHS

def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def generate_embeddings():
    # Load config.json
    with CONFIG_PATH.open("r") as f:
        config = json.load(f)

    for graph_config in config["graphs"]:
        try:
            graph_path = Path(graph_config["graph_path"])
            embedding_path = Path(graph_config["embedding_path"])
            model_path = Path(graph_config["model_path"])
            edge_embedding_path = Path(graph_config["edge_embedding_path"])

            # Ensure proper file extensions
            if not embedding_path.suffix == ".vec":
                embedding_path = embedding_path.with_suffix(".vec")
            if not model_path.suffix == ".model":
                model_path = model_path.with_suffix(".model")

            # Skip graph if all outputs already exist
            if embedding_path.exists() and model_path.exists() and edge_embedding_path.exists():
                print(f"Skipping {graph_path.name} (already processed)")
                continue

            print(f"\nProcessing graph: {graph_path.name}")

            # Load graph
            graph = nx.read_gml(graph_path)
            graph = nx.relabel_nodes(graph, lambda x: str(x))
            print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

            # Node2Vec
            n2v_model = Node2Vec(graph, dimensions=DIM, walk_length=WALK_LENGTH,
                                 num_walks=NUM_WALKS, workers=2)
            model = n2v_model.fit(window=WINDOW, min_count=1, batch_words=BATCH_WORDS)
            print("Node2Vec training completed")

            # Ensure directories exist
            ensure_dir(embedding_path)
            ensure_dir(model_path)
            ensure_dir(edge_embedding_path)

            # Save node embeddings and model
            model.wv.save_word2vec_format(str(embedding_path))
            model.save(str(model_path))
            print(f"Saved node embeddings: {embedding_path}")
            print(f"Saved Node2Vec model: {model_path}")

            # Edge embeddings
            model = Word2Vec.load(str(model_path))
            edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

            valid_edges = [(u, v) for u, v in graph.edges() if u in model.wv and v in model.wv]
            print(f"Generating edge embeddings for {len(valid_edges)} edges")

            with edge_embedding_path.open("w") as f:
                f.write(f"{len(valid_edges)} {DIM}\n")
                for i in tqdm(range(0, len(valid_edges), EDGE_BATCH_SIZE), desc="Edges batch"):
                    batch = valid_edges[i:i + EDGE_BATCH_SIZE]
                    for u, v in batch:
                        try:
                            emb = edges_embs[(u, v)]
                            line = f"{u}_{v} " + " ".join(map(str, emb)) + "\n"
                            f.write(line)
                        except KeyError:
                            continue

            print(f"Edge embeddings saved: {edge_embedding_path}")

        except Exception as e:
            print(f"Error processing {graph_config.get('graph_path', 'UNKNOWN')}: {e}")

    print("\nAll graphs processed successfully!")

if __name__ == "__main__":
    generate_embeddings()
