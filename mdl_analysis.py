"""
Minimum Description Length (MDL) analysis for graph partitions.

Weighted Poisson edge model:
  Edge cost     = [ W - Σ_{r,s} w_rs * log(w_rs / (n_r * n_s)) ] / ln2   (bits)
  Partition cost = [ logΓ(N+1) - Σ_r logΓ(n_r+1) ] / ln2                  (bits)
  Total DL       = Edge cost + Partition cost

All partitions are evaluated on the subgraph induced by GT-labeled nodes
(~95k nodes, same node set used in edge MI analysis).

Comparisons: GT (K≈743) vs WSBM (10 runs, K≈1083) vs SBM (50 runs, K≈1845)

Output: 3 figures
  1. Stacked bar  — GT / WSBM-mean / SBM-mean, broken into edge + partition cost
  2. WSBM runs    — individual DL per run (bar) + mean line
  3. Summary table
"""

import json
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ADJ_PATH      = "/Users/yuchi/Desktop/flywrite＿yuchi/sparse_connectivity_matrix.npz"
RID2IDX_PATH  = "/Users/yuchi/Desktop/flywrite/root_id_to_index_mapping_1.json"
GT_DICT_PATH  = "/Users/yuchi/Desktop/flywrite/root_id_type_dict.pkl"
WSBM_RUNS_DIR = "/Users/yuchi/Desktop/flywrite/runs_wsbm"

# ---------------------------------------------------------------------------
# MDL (weighted Poisson)
# ---------------------------------------------------------------------------
def compute_mdl(labels, src_loc, tgt_loc, wgt):
    """
    Compute weighted MDL for one partition.

    Parameters
    ----------
    labels   : (N,) int array — block assignment for each of the N nodes
    src_loc  : (E,) int array — source node local indices (0..N-1)
    tgt_loc  : (E,) int array — target node local indices
    wgt      : (E,) float     — edge weights (synapse counts)

    Returns
    -------
    total_bits, edge_bits, partition_bits : float
    """
    N = len(labels)
    unique_blocks, inv = np.unique(labels, return_inverse=True)
    K = len(unique_blocks)
    labels_c = inv                           # consecutive 0..K-1

    block_sizes = np.bincount(labels_c, minlength=K).astype(np.float64)   # (K,)

    # Weighted block-pair edge matrix  w_rs
    src_b = labels_c[src_loc]
    tgt_b = labels_c[tgt_loc]
    w_rs  = np.zeros((K, K), dtype=np.float64)
    np.add.at(w_rs, (src_b, tgt_b), wgt)

    nrns = np.outer(block_sizes, block_sizes)   # (K, K)
    W    = wgt.sum()

    # Edge cost (nats → bits)
    mask = w_rs > 0
    edge_nats = W - np.sum(w_rs[mask] * np.log(w_rs[mask] / nrns[mask]))
    edge_bits = edge_nats / np.log(2)

    # Partition cost (multinomial coefficient, bits)
    partition_bits = (gammaln(N + 1) - np.sum(gammaln(block_sizes + 1))) / np.log(2)

    return edge_bits + partition_bits, edge_bits, partition_bits


# ---------------------------------------------------------------------------
# Load adjacency matrix and GT labels
# ---------------------------------------------------------------------------
print("Loading adjacency matrix ...")
A = load_npz(ADJ_PATH)          # raw synapse counts

print("Loading GT labels ...")
with open(RID2IDX_PATH) as f:
    rid2idx = {int(k): int(v) for k, v in json.load(f).items()}

gt_dict = pickle.load(open(GT_DICT_PATH, "rb"))   # root_id → type_str
type_strings = sorted(set(gt_dict.values()))
type2cid     = {t: i for i, t in enumerate(type_strings)}

N_full = A.shape[0]
gt_label = np.full(N_full, -1, dtype=np.int32)
for rid, ctype in gt_dict.items():
    idx = rid2idx.get(rid, -1)
    if idx >= 0:
        gt_label[idx] = type2cid[ctype]

# GT-labeled node indices (matrix space)
gt_nodes = np.where(gt_label >= 0)[0]          # sorted array of valid indices
N_gt     = len(gt_nodes)
node2loc = {int(n): i for i, n in enumerate(gt_nodes)}   # matrix_idx → local_idx

# ---------------------------------------------------------------------------
# Build subgraph edges among GT-labeled nodes
# ---------------------------------------------------------------------------
print(f"Building subgraph for {N_gt:,} GT-labeled nodes ...")
coo  = A.tocoo()
src_all = coo.row.astype(np.int32)
tgt_all = coo.col.astype(np.int32)
wgt_all = coo.data.astype(np.float64)

valid = (gt_label[src_all] >= 0) & (gt_label[tgt_all] >= 0)
src_g = src_all[valid]
tgt_g = tgt_all[valid]
wgt_g = wgt_all[valid]

src_loc = np.array([node2loc[int(s)] for s in src_g], dtype=np.int32)
tgt_loc = np.array([node2loc[int(t)] for t in tgt_g], dtype=np.int32)

print(f"  Edges in subgraph: {len(src_loc):,}  |  Total synapses: {wgt_g.sum():,.0f}")

# GT labels for the N_gt nodes (local indexing)
gt_labels_loc = gt_label[gt_nodes]   # (N_gt,) int


# ---------------------------------------------------------------------------
# Helper: load cluster dict for any run → labels array (local indexing)
# ---------------------------------------------------------------------------
def load_labels_json(path):
    with open(path) as f:
        d = json.load(f)
    labels = np.full(N_gt, -1, dtype=np.int32)
    for rid, blk in d.items():
        idx = rid2idx.get(int(rid), -1)
        loc = node2loc.get(idx, -1)
        if loc >= 0:
            labels[loc] = int(blk)
    return labels

def mdl_from_labels(labels):
    """MDL on the GT-subgraph edges, ignoring nodes with label=-1."""
    valid = (labels[src_loc] >= 0) & (labels[tgt_loc] >= 0)
    good_nodes = labels >= 0
    labels_sub = labels[good_nodes]
    src_sub = src_loc[valid]
    tgt_sub = tgt_loc[valid]
    wgt_sub = wgt_g[valid]
    # remap local indices to sub-local indices
    old2new = np.full(N_gt, -1, dtype=np.int32)
    old2new[good_nodes] = np.arange(good_nodes.sum())
    src_sub2 = old2new[src_sub]
    tgt_sub2 = old2new[tgt_sub]
    return compute_mdl(labels_sub, src_sub2, tgt_sub2, wgt_sub)


# ---------------------------------------------------------------------------
# GT MDL
# ---------------------------------------------------------------------------
print("\nComputing GT MDL ...")
gt_total, gt_edge, gt_part = mdl_from_labels(gt_labels_loc)
gt_K = len(np.unique(gt_labels_loc[gt_labels_loc >= 0]))
print(f"  GT  K={gt_K:,}  edge={gt_edge:.4e}  partition={gt_part:.4e}  total={gt_total:.4e} bits")

# ---------------------------------------------------------------------------
# WSBM MDL (10 runs)
# ---------------------------------------------------------------------------
print("\nComputing WSBM MDL (10 runs) ...")
wsbm_results = []
wsbm_Ks = []
for run in sorted(os.listdir(WSBM_RUNS_DIR)):
    if not run.startswith("sbm_"):
        continue
    path = os.path.join(WSBM_RUNS_DIR, run, "cluster_assignment_dict.json")
    labels = load_labels_json(path)
    tot, edg, prt = mdl_from_labels(labels)
    K = len(np.unique(labels[labels >= 0]))
    wsbm_results.append((tot, edg, prt))
    wsbm_Ks.append(K)
    print(f"  {run}: K={K}  total={tot:.4e} bits")

wsbm_arr   = np.array(wsbm_results)   # (10, 3)
wsbm_mean  = wsbm_arr.mean(axis=0)
wsbm_std   = wsbm_arr.std(axis=0)
wsbm_K_mean = np.mean(wsbm_Ks)

# ---------------------------------------------------------------------------
# Figure 1 — Stacked bar: GT / WSBM-mean / SBM-mean
# ---------------------------------------------------------------------------
print("\nPlotting Figure 1: Stacked bar ...")
fig1, ax = plt.subplots(figsize=(8, 6))

models  = ['GT', 'WSBM\n(mean)']
edge_v  = [gt_edge,   wsbm_mean[1]]
part_v  = [gt_part,   wsbm_mean[2]]
total_e = [0,         wsbm_std[0]]

x = np.arange(len(models))
w = 0.5

b1 = ax.bar(x, edge_v, width=w, label='Edge cost',      color='#4C72B0', alpha=0.85)
b2 = ax.bar(x, part_v, width=w, label='Partition cost', color='#DD8452', alpha=0.85,
            bottom=edge_v,
            yerr=total_e, capsize=6, error_kw=dict(ecolor='black', lw=1.5))

# annotate totals
for i, (ev, pv, ee) in enumerate(zip(edge_v, part_v, total_e)):
    total = ev + pv
    offset = ee + total * 0.02
    ax.text(x[i], total + offset, f'{total:.2e}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('Description Length (bits)', fontsize=11)
ax.set_title('MDL Comparison: GT vs WSBM vs SBM\n(weighted Poisson, GT-node subgraph)', fontsize=11)
ax.legend(fontsize=10)
ax.yaxis.get_major_formatter().set_scientific(True)
ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
plt.tight_layout()
plt.savefig('mdl_stacked_bar.png', dpi=150, bbox_inches='tight')
print("  Saved → mdl_stacked_bar.png")

# ---------------------------------------------------------------------------
# Figure 2 — WSBM runs variance
# ---------------------------------------------------------------------------
print("Plotting Figure 2: WSBM runs variance ...")
fig2, ax = plt.subplots(figsize=(9, 5))

run_labels = [f'wsbm_{i}' for i in range(len(wsbm_results))]
run_totals = wsbm_arr[:, 0]
run_edges  = wsbm_arr[:, 1]
run_parts  = wsbm_arr[:, 2]
x2 = np.arange(len(run_labels))

ax.bar(x2, run_edges, label='Edge cost',      color='#4C72B0', alpha=0.85)
ax.bar(x2, run_parts, label='Partition cost', color='#DD8452', alpha=0.85, bottom=run_edges)
ax.axhline(wsbm_mean[0], color='black', linestyle='--', linewidth=1.5,
           label=f'Mean = {wsbm_mean[0]:.3e}')
ax.fill_between([-0.5, len(run_labels) - 0.5],
                wsbm_mean[0] - wsbm_std[0],
                wsbm_mean[0] + wsbm_std[0],
                color='gray', alpha=0.15, label=f'±1 std ({wsbm_std[0]:.2e})')

ax.set_xticks(x2)
ax.set_xticklabels(run_labels, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Description Length (bits)', fontsize=11)
ax.set_title('WSBM MDL per Run (weighted Poisson)', fontsize=11)
ax.legend(fontsize=9)
ax.yaxis.get_major_formatter().set_scientific(True)
ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
plt.tight_layout()
plt.savefig('mdl_wsbm_runs.png', dpi=150, bbox_inches='tight')
print("  Saved → mdl_wsbm_runs.png")

# ---------------------------------------------------------------------------
# Figure 3 — Summary table
# ---------------------------------------------------------------------------
print("Plotting Figure 3: Summary table ...")
fig3, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

def fmt(v, e=None):
    s = f'{v:.4e}'
    if e is not None and e > 0:
        s += f' ± {e:.2e}'
    return s

rows = [
    ['Model',        'K',                  'Edge cost (bits)',              'Partition cost (bits)',        'Total DL (bits)'],
    ['GT',           str(gt_K),            fmt(gt_edge),                   fmt(gt_part),                   fmt(gt_total)],
    ['WSBM (mean)',  f'{wsbm_K_mean:.0f}', fmt(wsbm_mean[1], wsbm_std[1]), fmt(wsbm_mean[2], wsbm_std[2]), fmt(wsbm_mean[0], wsbm_std[0])],
]

tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
               cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width(list(range(len(rows[0]))))
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2d2d2d')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 1:
        cell.set_facecolor('#f0f4ff')
    cell.set_edgecolor('#cccccc')

ax.set_title('MDL Summary (weighted Poisson, GT-node subgraph)', fontsize=11, pad=12)
plt.tight_layout()
plt.savefig('mdl_summary_table.png', dpi=150, bbox_inches='tight')
print("  Saved → mdl_summary_table.png")

plt.close('all')
print("\nDone.")
