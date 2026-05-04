# Information Theory and Cognition — FlyWire Connectome Analysis

Clustering and information-theoretic analysis of neurons from the *Drosophila* connectome ([FlyWire v783](https://flywire.ai)).  
We fit a **Weighted Stochastic Block Model (WSBM)** to the synaptic connectivity matrix and evaluate the learned partition against ground-truth (GT) cell-type annotations using mutual information and minimum description length.

---

## Repository Structure

```
.
├── connectivity_matrix_construction.py   # Step 1 — build sparse connectivity matrix
├── index_mapping.py                      # Utility — root_id ↔ matrix index
├── sbm_wsbm_space.py                     # Step 2 — WSBM hyperparameter search (training)
├── hidden_markov_graph_wsbm.py           # Step 3 — WSBM inference / save cluster assignments
├── wsbm_hinton.py                        # Analysis — Hinton diagram (GT vs WSBM)
├── wsbm_credible_interval.py             # Analysis — credible interval + random clustering baseline
├── edge_mi_analysis.py                   # Analysis — weighted edge mutual information
├── mdl_analysis.py                       # Analysis — minimum description length
├── new_cluster_similarity_test.ipynb     # Analysis — Hungarian matching & cluster similarity
├── requirements.txt
├── root_id_to_index_mapping_1.json       # Precomputed — node index mapping (138,639 neurons)
├── root_id_type_dict.pkl                 # Precomputed — GT cell-type annotations (95,145 neurons)
├── runs_wsbm/                            # Precomputed — WSBM cluster assignments (10 runs)
│   └── sbm_{0..9}/cluster_assignment_dict.json
└── wsbm_best_match/                      # Precomputed — Hungarian-matched best clusters (10 runs)
    └── wsbm_best_match_{0..9}.csv
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.x recommended. GPU optional (training runs on CPU by default).

---

## Data

Download the raw FlyWire v783 connectome data:

| File | URL |
|---|---|
| Synaptic connections (unfiltered) | https://codex.flywire.ai/api/download?data_product=connections_no_threshold&data_version=783 |
| Neuron names / cell types | https://codex.flywire.ai/api/download?data_product=names&data_version=783 |

---

## Pipeline

### Step 1 — Build connectivity matrix

Takes the raw connections CSV and produces a sparse weighted adjacency matrix (synapse counts) and the root-id-to-index mapping.

```bash
python connectivity_matrix_construction.py
```

**Input:** `connections_no_threshold.csv.gz`  
**Output:** `sparse_connectivity_matrix_count.npz`, `root_id_to_index_mapping_1.json`

---

### Step 2 — WSBM hyperparameter search (optional — precomputed results included)

Bayesian hyperparameter search over K, learning rate, optimizer using Gaussian Process.

```bash
python sbm_wsbm_space.py
```

**Input:** `sparse_connectivity_matrix_count.npz`  
**Output:** best hyperparameters printed to stdout

---

### Step 3 — WSBM inference

Run WSBM with the best hyperparameters to produce cluster assignments.  
Results are saved to `runs_wsbm/sbm_{task_id}/`. Precomputed results for 10 runs are already included in this repo.

```bash
python hidden_markov_graph_wsbm.py
```

**Input:** `sparse_connectivity_matrix_count.npz`, `root_id_to_index_mapping_1.json`  
**Output:** `runs_wsbm/sbm_{id}/cluster_assignment_dict.json` (and model weights)

---

## Analysis

All analyses below use the **precomputed cluster assignments** in `runs_wsbm/` and the GT annotations in `root_id_type_dict.pkl`. The connectivity matrix (`sparse_connectivity_matrix_count.npz`) is also needed for edge MI and MDL.

### Cluster similarity & Hungarian matching

```bash
jupyter notebook new_cluster_similarity_test.ipynb
```

Computes pairwise cluster similarity between WSBM and GT using Hungarian algorithm.

---

### Hinton diagram

Visualises the confusion matrix between top-50 GT types and their best-matched WSBM clusters.

```bash
python wsbm_hinton.py
```

**Output:** `wsbm_hinton.png`

---

### Credible interval + random clustering baseline

Bootstraps a credible interval for WSBM cluster-matching score and compares against a random partition baseline.

```bash
python wsbm_credible_interval.py
```

**Output:** `wsbm_credible_interval.png`

---

### Weighted Edge Mutual Information

Computes MI and NMI between source/target cluster label pairs, weighted by synapse counts.  
Compares GT↔GT, WSBM↔WSBM, and GT↔WSBM.

```bash
python edge_mi_analysis.py
```

**Input:** `sparse_connectivity_matrix_count.npz`, `root_id_to_index_mapping_1.json`, `root_id_type_dict.pkl`, `runs_wsbm/sbm_6/cluster_assignment_dict.json`  
**Output:** `edge_mi_results.png`

---

### Minimum Description Length (MDL)

Evaluates how compactly each partition (GT vs WSBM) describes the graph using a weighted Poisson model.  
All comparisons restricted to the GT-labeled node subgraph (~95k nodes).

```bash
python mdl_analysis.py
```

**Input:** `sparse_connectivity_matrix_count.npz`, `root_id_to_index_mapping_1.json`, `root_id_type_dict.pkl`, `runs_wsbm/`  
**Output:** `mdl_stacked_bar.png`, `mdl_wsbm_runs.png`, `mdl_summary_table.png`

---

## Key Results

| Metric | GT | WSBM (mean) |
|---|---|---|
| NMI (weighted edge MI) | 0.2697 | 0.2664 |
| Total MDL (bits) | 7.207e+07 | 6.863e+07 |
| K (clusters, GT-node subgraph) | 743 | ~737 |

WSBM achieves near-identical NMI to GT cell types while compressing the graph ~3.4% more efficiently under MDL.

---

## Reference

[Stochastic variational inference for low-rank stochastic block models](https://kyunghyuncho.me/stochastic-variational-inference-for-low-rank-stochastic-block-models-or-how-i-re-discovered-sbm-unnecessarily/)
