import numpy as np
import pickle
import json
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

# ── Hinton diagram ─────────────────────────────────────────────────────────
def hinton(matrix, max_weight=None, ax=None):
    ax = ax or plt.gca()
    if max_weight is None:
        max_weight = matrix.max() if matrix.max() > 0 else 1

    ax.patch.set_facecolor('#AAAAAA')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (row, col), val in np.ndenumerate(matrix):
        if val == 0:
            continue
        size = np.sqrt(val / max_weight)
        rect = patches.FancyBboxPatch(
            [col - size / 2, row - size / 2], size, size,
            boxstyle="square,pad=0",
            facecolor='white', edgecolor='white'
        )
        ax.add_patch(rect)

    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(-0.5, matrix.shape[0] - 0.5)
    ax.invert_yaxis()

# ── load GT ────────────────────────────────────────────────────────────────
root_id_type_dict = pickle.load(open('root_id_type_dict.pkl', 'rb'))
gt_types = list(set(root_id_type_dict.values()))
type_to_cid = {t: i for i, t in enumerate(gt_types)}

# ── load WSBM (use sbm_0) ──────────────────────────────────────────────────
run = 'runs_wsbm/sbm_0'
with open(f'{run}/cluster_assignment_dict.json') as f:
    wsbm_dict = json.load(f)

shared = list(set(root_id_type_dict.keys())
              .intersection({int(k) for k in wsbm_dict.keys()}))

gt_labels   = np.array([type_to_cid[root_id_type_dict[rid]] for rid in shared])
wsbm_labels = np.array([wsbm_dict[str(rid)] for rid in shared])

gt_type_names = np.array(gt_types)   # index → type name

# ── select top N GT clusters by size ──────────────────────────────────────
TOP_N = 50
gt_counts = Counter(gt_labels)
top_gt_cids = [cid for cid, _ in gt_counts.most_common(TOP_N)]
top_gt_names = [gt_type_names[cid] for cid in top_gt_cids]

# ── build confusion matrix for top N GT clusters ───────────────────────────
# rows = top GT clusters, cols = all WSBM clusters
wsbm_unique = np.unique(wsbm_labels)
wsbm_to_col = {w: i for i, w in enumerate(wsbm_unique)}

cm = np.zeros((TOP_N, len(wsbm_unique)), dtype=np.float32)
for gt_c, ws_c in zip(gt_labels, wsbm_labels):
    if gt_c in set(top_gt_cids):
        row = top_gt_cids.index(gt_c)
        col = wsbm_to_col[ws_c]
        cm[row, col] += 1

# ── Hungarian alignment: match each GT row to best WSBM column ─────────────
row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
# reorder columns so best matches come first (aligned to rows)
aligned_cm = cm[:, col_ind]

# WSBM cluster IDs in the aligned order
aligned_wsbm_ids = wsbm_unique[col_ind]
# label WSBM columns with the GT type they were matched to
aligned_wsbm_labels = [top_gt_names[r] for r in row_ind]

# Pad to square: col_ind may have fewer entries than TOP_N if TOP_N > len(wsbm_unique)
# In practice wsbm_unique >> TOP_N so aligned_cm is TOP_N x TOP_N
aligned_cm_square = aligned_cm[:, :TOP_N]
aligned_wsbm_labels_square = aligned_wsbm_labels[:TOP_N]

print(f"Confusion matrix shape: {aligned_cm_square.shape}")
print(f"GT top types (first 5): {top_gt_names[:5]}")
print(f"WSBM matched labels (first 5): {aligned_wsbm_labels_square[:5]}")

# ── plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 12))

hinton(aligned_cm_square, ax=ax)

ax.set_xticks(np.arange(TOP_N))
ax.set_yticks(np.arange(TOP_N))
ax.set_xticklabels(aligned_wsbm_labels_square, fontsize=6, rotation=90, ha='right')
ax.set_yticklabels(top_gt_names, fontsize=6)

ax.set_xlabel('WSBM cluster (labeled with matched GT cell type)', fontsize=10)
ax.set_ylabel('Ground Truth cell type', fontsize=10)
ax.set_title(f'WSBM vs GT — Hinton Diagram (Top {TOP_N} GT cell types by size)\n'
             f'Hungarian-aligned | square size ∝ # neurons', fontsize=12)

plt.tight_layout()
plt.savefig('wsbm_hinton.png', dpi=150)
plt.show()
print("Saved → wsbm_hinton.png")
