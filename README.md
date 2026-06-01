# Detecting Android Malware Using Graph Neural Networks

This repository contains the implementation used for the bachelor thesis:

**Detecting Android Malware Using Graph Neural Networks: A Comparison Between an Inductive and Transductive Learning Approach**

The project compares a transductive Graph Convolutional Network (GCN) and an inductive GraphSAGE model for Android malware detection using application call graphs generated from APK files.

Leiden University
BSc Data Science & Artificial Intelligence

---

Author

Priya G. A. Smit

2026

---

# Requirements

This project uses two separate Python environments.

The Android preprocessing pipeline relies on Androguard and Node2Vec running in a Python 3.7 environment, while model training and explainability were performed in a separate Python 3.10 Conda environment.

---

## Environment 1: APK Processing and Node2Vec

Used for:

* APK download
* APK decompilation
* Call graph generation
* Node2Vec embedding generation

Python version:

```text
Python 3.7.13
```

Main packages:

| Package         | Version |
| --------------- | ------- |
| androguard      | 3.3.5   |
| node2vec        | 0.4.3   |
| networkx        | 2.5     |
| gensim          | 4.2.0   |
| numpy           | 1.21.6  |
| pandas          | 1.0.5   |
| scikit-learn    | 0.22.2  |
| torch           | 1.13.1  |
| torch-geometric | 2.3.1   |


---

## Environment 2: GNN Training and Explainability

Used for:

* Dataset construction
* Model training
* Model evaluation
* Explainability analysis

Python version:

```text
Python 3.10.19
```

Main packages:

| Package         | Version |
| --------------- | ------- |
| torch           | 2.1.0   |
| torch-geometric | 2.7.0   |
| networkx        | 3.4.2   |
| numpy           | 1.26.4  |
| scipy           | 1.15.3  |
| matplotlib      | 3.10.8  |
| tqdm            | 4.67.1  |

---

## External Software

The following software must be installed separately:

* Apktool
* Conda

Apktool is required for APK decompilation.

---

# Repository Structure

```text
project/
│
├── download_androzoo_1000.py
├── decompile_apk.py
├── generate_config.py
├── generate_cg.py
├── embedding.py
├── build_gnn_dataset.py
├── dataset_pyg.py
├── models_pyg.py
├── train_eval_pyg.py
├── view_explanations.py
│
├── apks/
├── decompiled/
├── callgraphs/
├── embeddings/
├── gnn_preprocessed/
└── runs_pyg/
```

---

# Pipeline Overview

The complete workflow consists of eight stages:

```text
APK Download
      ↓
APK Decompilation
      ↓
Configuration Generation
      ↓
Call Graph Generation
      ↓
Node2Vec Embeddings
      ↓
Dataset Construction
      ↓
Model Training
      ↓
Explainability Evaluation
```

# Step 1: Download APKs

Download Android applications from AndroZoo.

```bash
python3 download_androzoo_1000.py --csv apk.csv --apikey "YOUR_ANDROZOO_API_KEY" --balanced --n 1000 --sleep 1.0
```

Output:

```text
apks/
├── app1.apk
├── app2.apk
└── ...
```

---

# Step 2: Decompile APKs

Decompile downloaded APK files using Apktool.

```bash
python decompile_apk.py
```

Output:

```text
decompiled/
├── app1/
├── app2/
└── ...
```

---

# Step 3: Generate Configuration File

Generate a configuration file containing APK locations and labels.

```bash
python generate_config.py
```

This file is used by the subsequent preprocessing stages.

---

# Step 4: Generate Call Graphs

Construct application call graphs using Androguard.

```bash
python generate_cg.py
```

Output:

```text
callgraphs/
├── app1.gml
├── app2.gml
└── ...
```

Each node represents a method and each edge represents a method invocation.

---

# Step 5: Generate Node2Vec Embeddings

Generate Node2Vec embeddings for all call graph nodes.

```bash
python embedding.py
```

Output:

```text
embeddings/
├── app1.npy
├── app2.npy
└── ...
```

Default embedding dimension:

```text
32
```

---

# Step 6: Build the GNN Dataset

Switch to the training environment:

Combine call graphs, node embeddings, and labels into PyTorch Geometric datasets.

```bash
python build_gnn_dataset.py
```

Output:

```text
gnn_preprocessed/
├── app_graph.pkl
├── app_features.npy
├── app_label.json
└── ...
```

Each graph contains:

* Graph structure
* Node feature matrix
* Binary malware label

---

# Step 7: Train a Model

## Train GCN

```bash
python train_eval_pyg.py --model gcn --data_dir gnn_preprocessed --epochs 80
```

## Train GraphSAGE

```bash
python train_eval_pyg.py --model sage --data_dir gnn_preprocessed --epochs 80 
```

---

# Step 8: Explainability Analysis

Generate explanations using GNNExplainer.

Example:

```bash
python train_eval_pyg.py --model gcn --data_dir gnn_preprocessed --epochs 80 --explain --explain_n 40 --explain_confidence_min 0.60
```

Optional parameters:

```text
--explain
--explain_n
--explain_focus
--explain_confidence_min
--explain_epochs
```

Generated explanations are stored in:

```text
runs_pyg/
└── model_name/
    └── explanations/
```

# Models

## Graph Convolutional Network (GCN)

Architecture:

* 2 Graph Convolution layers
* Batch Normalization
* ReLU activation
* Global Mean Pooling
* Dropout
* Linear classifier

Learning type:

```text
Transductive
```

---

## GraphSAGE

Architecture:

* 2 SAGEConv layers
* Batch Normalization
* ReLU activation
* Global Mean Pooling
* Dropout
* Linear classifier

Learning type:

```text
Inductive
```

---

## Classification Task

Both models perform graph-level binary classification:

```text
0 = Benign
1 = Malicious
```

---

# Output

After training, results are stored in:

```text
runs_pyg/
├── gcn_seed123/
└── sage_seed123/
```

Reported metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix Statistics

---

## Step 9: Plotting Generated Explanations

GraphSAGE:

```bash
python view_explanations.py --exp_dir runs_pyg/sage_seed123/explanations --topk 10
```

GCN:

```bash
python view_explanations.py --exp_dir runs_pyg/gcn_seed123/explanations --topk 10
```

Optional parameters:

```text
--topk
--file
```

Generated explanations are stored in:

```text
plots/
├── confusion_matrix_gcn.png
├── confusion_matrix_graphsage.png
├── results_summary.csv
├── test_metric_bars.png
├── unfaithfulness_boxplot.png
└── unfaithfulness_hist.png
```
---

# Reproducibility Notes

The preprocessing and training stages were intentionally separated into two environments because of dependency conflicts between the Android analysis tooling (Androguard/Node2Vec) and the more recent PyTorch Geometric ecosystem used for model training and explainability.

To reproduce the thesis results, execute the pipeline sequentially from Step 1 through Step 9, switching environments after Node2Vec embedding generation.
