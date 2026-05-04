"""
Edge-level Mutual Information analysis (weighted by synapse counts).

For every edge (i -> j) in the connectivity matrix, we record the label pair
(label[i], label[j]) and compute weighted MI from the resulting joint distribution,
where each edge is weighted by its synapse count.

Three comparisons:
  1. MI( GT_type[src],    GT_type[tgt]   )  — GT grouping explains connectivity?
  2. MI( WSBM_block[src], WSBM_block[tgt] ) — WSBM grouping explains connectivity?
  3. MI( GT_type[src],    WSBM_block[tgt] ) — cross: does GT src predict WSBM tgt?

All three share the same edge set: only edges where BOTH endpoints have a GT label
AND a WSBM block assignment.
"""

import numpy as np
import json
import pickle
import zipfile
from scipy.sparse import load_npz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths — adjust if needed
# ---------------------------------------------------------------------------
ADJ_PATH      = "/Users/yuchi/Desktop/flywrite＿yuchi/sparse_connectivity_matrix.npz"
RID2IDX_PATH  = "/Users/yuchi/Desktop/flywrite/root_id_to_index_mapping_1.json"
GT_DICT_PATH  = "/Users/yuchi/Desktop/flywrite/root_id_type_dict.pkl"
WSBM_ZIP_PATH = "/Users/yuchi/Desktop/flywrite/sbm_6.zip"
MODEL_LABEL   = "WSBM"


# ---------------------------------------------------------------------------
# Weighted MI
# ---------------------------------------------------------------------------
def mi_from_pairs(labels_a, labels_b, weights=None):
    """Weighted MI(A; B). weights = synapse counts per edge."""
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if weights is None:
        weights = np.ones(len(labels_a), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    def _remap(arr):
        uniq = np.unique(arr)
        m = {v: i for i, v in enumerate(uniq)}
        return np.array([m[v] for v in arr]), len(uniq)

    a, Ka = _remap(labels_a)
    b, Kb = _remap(labels_b)

    joint_w = np.zeros((Ka, Kb), dtype=np.float64)
    np.add.at(joint_w, (a, b), weights)
    total = joint_w.sum()

    pxy = joint_w / total
    px  = pxy.sum(axis=1)
    py  = pxy.sum(axis=0)

    outer = np.outer(px, py)
    mask  = pxy > 0
    return np.sum(pxy[mask] * np.log(pxy[mask] / outer[mask]))


def nmi_from_pairs(labels_a, labels_b, weights=None):
    """Weighted NMI: MI / sqrt(H(A) * H(B))."""
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if weights is None:
        weights = np.ones(len(labels_a), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    def _remap(arr):
        uniq = np.unique(arr)
        m = {v: i for i, v in enumerate(uniq)}
        return np.array([m[v] for v in arr])

    def _weighted_entropy(arr, w):
        pw = np.zeros(arr.max() + 1)
        np.add.at(pw, arr, w)
        pw /= pw.sum()
        pw = pw[pw > 0]
        return -np.sum(pw * np.log(pw))

    a = _remap(labels_a)
    b = _remap(labels_b)

    mi = mi_from_pairs(a, b, weights)
    ha = _weighted_entropy(a, weights)
    hb = _weighted_entropy(b, weights)
    denom = np.sqrt(ha * hb)
    return mi / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading adjacency matrix ...")
A = load_npz(ADJ_PATH)
# keep raw synapse counts for weighted MI

print("Loading mappings ...")
with open(RID2IDX_PATH) as f:
    rid2idx = {int(k): int(v) for k, v in json.load(f).items()}

gt_dict = pickle.load(open(GT_DICT_PATH, "rb"))   # root_id -> cell_type_str
with zipfile.ZipFile(WSBM_ZIP_PATH) as z:
    with z.open("cluster_assignment_dict.json") as f:
        sbm_dict = {int(k): int(v) for k, v in json.load(f).items()}  # root_id -> block_id

type_strings = sorted(set(gt_dict.values()))
type2cid = {t: i for i, t in enumerate(type_strings)}

# ---------------------------------------------------------------------------
# Build per-matrix-index label arrays
# ---------------------------------------------------------------------------
N = A.shape[0]
gt_label  = np.full(N, -1, dtype=np.int32)
sbm_label = np.full(N, -1, dtype=np.int32)

for rid, ctype in gt_dict.items():
    idx = rid2idx.get(rid, -1)
    if idx >= 0:
        gt_label[idx] = type2cid[ctype]

for rid, block in sbm_dict.items():
    idx = rid2idx.get(rid, -1)
    if idx >= 0:
        sbm_label[idx] = int(block)

# ---------------------------------------------------------------------------
# Extract edges where BOTH endpoints have valid GT AND WSBM labels
# ---------------------------------------------------------------------------
print("Extracting edge list ...")
coo = A.tocoo()
src = coo.row.astype(np.int32)
tgt = coo.col.astype(np.int32)
wgt = coo.data.astype(np.float64)   # synapse counts

valid = (
    (gt_label[src]  >= 0) &
    (gt_label[tgt]  >= 0) &
    (sbm_label[src] >= 0) &
    (sbm_label[tgt] >= 0)
)
src = src[valid]
tgt = tgt[valid]
wgt = wgt[valid]
print(f"  Edges used: {len(src):,}  (out of {coo.nnz:,} total)")
print(f"  Total synapses: {wgt.sum():,.0f}")

gt_src  = gt_label[src]
gt_tgt  = gt_label[tgt]
sbm_src = sbm_label[src]
sbm_tgt = sbm_label[tgt]

# ---------------------------------------------------------------------------
# Compute weighted MIs
# ---------------------------------------------------------------------------
print("\nComputing MI(GT_src, GT_tgt) ...")
mi_gt_gt   = mi_from_pairs(gt_src,  gt_tgt,  weights=wgt)
nmi_gt_gt  = nmi_from_pairs(gt_src, gt_tgt,  weights=wgt)

print(f"Computing MI({MODEL_LABEL}_src, {MODEL_LABEL}_tgt) ...")
mi_sbm_sbm  = mi_from_pairs(sbm_src,  sbm_tgt,  weights=wgt)
nmi_sbm_sbm = nmi_from_pairs(sbm_src, sbm_tgt,  weights=wgt)

print(f"Computing MI(GT_src, {MODEL_LABEL}_tgt) ...")
mi_gt_sbm   = mi_from_pairs(gt_src,  sbm_tgt,  weights=wgt)
nmi_gt_sbm  = nmi_from_pairs(gt_src, sbm_tgt,  weights=wgt)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
ratio = nmi_sbm_sbm / nmi_gt_gt
print("\n" + "="*55)
print(f"{'Metric':<35} {'MI (nats)':>10}  {'NMI':>6}")
print("-"*55)
print(f"{'MI(GT_src, GT_tgt)':<35} {mi_gt_gt:>10.4f}  {nmi_gt_gt:>6.4f}")
print(f"{f'MI({MODEL_LABEL}_src, {MODEL_LABEL}_tgt)':<35} {mi_sbm_sbm:>10.4f}  {nmi_sbm_sbm:>6.4f}")
print(f"{f'MI(GT_src, {MODEL_LABEL}_tgt)':<35} {mi_gt_sbm:>10.4f}  {nmi_gt_sbm:>6.4f}")
print("="*55)
print(f"\n{MODEL_LABEL} / GT ratio (NMI): {ratio:.4f}")

# ---------------------------------------------------------------------------
# Plot — single figure: NMI bar + MI bar + summary table
# ---------------------------------------------------------------------------
print("\nPlotting results ...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Weighted Edge MI Analysis ({MODEL_LABEL})", fontsize=14, fontweight='bold')

nmi_vals   = [nmi_gt_gt, nmi_sbm_sbm, nmi_gt_sbm]
mi_vals    = [mi_gt_gt,  mi_sbm_sbm,  mi_gt_sbm]
xlabels    = ['GT↔GT', f'{MODEL_LABEL}↔{MODEL_LABEL}', f'GT↔{MODEL_LABEL}']
colors     = ['#4C72B0', '#DD8452', '#55A868']

# NMI bar
ax = axes[0]
bars = ax.bar(xlabels, nmi_vals, color=colors, edgecolor='white', width=0.5)
for bar, val in zip(bars, nmi_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(nmi_vals) * 1.3)
ax.set_ylabel('NMI')
ax.set_title('NMI Comparison (weighted)')

# MI bar
ax = axes[1]
bars = ax.bar(xlabels, mi_vals, color=colors, edgecolor='white', width=0.5)
for bar, val in zip(bars, mi_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(mi_vals) * 1.3)
ax.set_ylabel('MI (nats)')
ax.set_title('MI Comparison (weighted)')

# Summary table
ax = axes[2]
ax.axis('off')
table_data = [
    ['Metric',                        'MI (nats)',          'NMI'],
    ['GT↔GT',                         f'{mi_gt_gt:.4f}',    f'{nmi_gt_gt:.4f}'],
    [f'{MODEL_LABEL}↔{MODEL_LABEL}',  f'{mi_sbm_sbm:.4f}',  f'{nmi_sbm_sbm:.4f}'],
    [f'GT↔{MODEL_LABEL}',             f'{mi_gt_sbm:.4f}',   f'{nmi_gt_sbm:.4f}'],
    [f'{MODEL_LABEL}/GT ratio',       '—',                   f'{ratio:.4f}'],
    ['Edges',                         f'{len(src):,}',       '—'],
    ['Total synapses',                f'{wgt.sum():,.0f}',   '—'],
]
tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               cellLoc='center', loc='center', bbox=[0, 0.05, 1, 0.9])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2d2d2d')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#f0f4ff')
    cell.set_edgecolor('#cccccc')
ax.set_title('Summary', fontsize=12, pad=10)

plt.tight_layout()
plt.savefig('edge_mi_results.png', dpi=150, bbox_inches='tight')
print("  Saved → edge_mi_results.png")
