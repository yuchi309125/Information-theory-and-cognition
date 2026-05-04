import numpy as np
import json
import pickle
import os
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# ── helpers (same as cluster_similarity_test) ──────────────────────────────
def confusion_matrix(X, Y):
    unique_x = np.unique(X)
    unique_y = np.unique(Y)
    x_dict = {v: i for i, v in enumerate(unique_x)}
    y_dict = {v: i for i, v in enumerate(unique_y)}
    cm = np.zeros((len(unique_x), len(unique_y)))
    for xi, yi in zip(X, Y):
        cm[x_dict[xi], y_dict[yi]] += 1
    return cm

def compare_two_assignments(A, B):
    cm = confusion_matrix(A, B)
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    score = cm[row_ind, col_ind].sum()
    return score, cm[:, col_ind]

# ── ground truth labels ────────────────────────────────────────────────────
root_id_type_dict = pickle.load(open('root_id_type_dict.pkl', 'rb'))
cluster_strings = list(set(root_id_type_dict.values()))
cluster_string_to_cid = {s: i for i, s in enumerate(cluster_strings)}

# ── wsbm scores ────────────────────────────────────────────────────────────
runs_dir = 'runs_wsbm'
run_dirs = sorted([d for d in os.listdir(runs_dir) if d.startswith('sbm_')])
print(f"Found {len(run_dirs)} wsbm runs: {run_dirs}")

scores_wsbm = []
for run in run_dirs:
    path = os.path.join(runs_dir, run, 'cluster_assignment_dict.json')
    with open(path) as f:
        wsbm_dict = json.load(f)
    shared = list(set(root_id_type_dict.keys())
                  .intersection({int(k) for k in wsbm_dict.keys()}))
    gt  = np.array([cluster_string_to_cid[root_id_type_dict[rid]] for rid in shared])
    est = np.array([wsbm_dict[str(rid)] for rid in shared])
    score, _ = compare_two_assignments(gt, est)
    scores_wsbm.append(score)
    print(f"  {run}: score={score:.0f}  shared={len(shared)}")

scores_wsbm = np.array(scores_wsbm)

# use last run's gt/shared for baselines
N_ITER = 100
n_gt_classes = len(np.unique(gt))
n = len(gt)

# ── ground truth CI via bootstrap (GT vs GT) ──────────────────────────────
print("\nComputing GT bootstrap CI...")
scores_gt = []
for _ in range(N_ITER):
    idx = np.random.choice(n, size=n, replace=True)
    gt_boot = gt[idx]
    s, _ = compare_two_assignments(gt_boot, gt_boot)
    scores_gt.append(s)
scores_gt = np.array(scores_gt)

# ── random baselines ───────────────────────────────────────────────────────
print("Computing random baselines...")
scores_opt  = []
scores_pess = []
for _ in range(N_ITER):
    rand_opt  = np.random.permutation(gt)
    rand_pess = np.random.randint(0, n_gt_classes, n)
    s_opt,  _ = compare_two_assignments(gt, rand_opt)
    s_pess, _ = compare_two_assignments(gt, rand_pess)
    scores_opt.append(s_opt)
    scores_pess.append(s_pess)

scores_opt  = np.array(scores_opt)
scores_pess = np.array(scores_pess)

# ── statistics ─────────────────────────────────────────────────────────────
def stats(s):
    return s.mean(), *np.percentile(s, [2.5, 97.5])

mean_g, lo_g, hi_g   = stats(scores_gt)
mean_w, lo_w, hi_w   = stats(scores_wsbm)
mean_o, lo_o, hi_o   = stats(scores_opt)
mean_p, lo_p, hi_p   = stats(scores_pess)

print(f"\nGround Truth — mean={mean_g:.1f}  95% CI [{lo_g:.1f}, {hi_g:.1f}]")
print(f"WSBM         — mean={mean_w:.1f}  95% CI [{lo_w:.1f}, {hi_w:.1f}]")
print(f"Random opt   — mean={mean_o:.1f}  95% CI [{lo_o:.1f}, {hi_o:.1f}]")
print(f"Random pess  — mean={mean_p:.1f}  95% CI [{lo_p:.1f}, {hi_p:.1f}]")

# ── plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

data      = [scores_gt, scores_wsbm, scores_opt, scores_pess]
labels    = ['Ground Truth', 'WSBM', 'Random\n(optimistic)', 'Random\n(pessimistic)']
means     = [mean_g, mean_w, mean_o, mean_p]
cis       = [(lo_g, hi_g), (lo_w, hi_w), (lo_o, hi_o), (lo_p, hi_p)]
colors    = ['lightgreen', 'lightblue', 'lightsalmon', 'lavender']
positions = [1, 2, 3, 4]

box = ax.boxplot(data, patch_artist=True, positions=positions, widths=0.5)
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

ax.scatter(positions, means, marker='D', color='red', s=80, zorder=4, label='Mean')

for pos, (lo, hi), m in zip(positions, cis, means):
    ax.errorbar(pos, m, yerr=[[m - lo], [hi - m]],
                fmt='none', color='black', capsize=6, linewidth=1.5)

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel('Score (Hungarian-matched)', fontsize=12)
ax.set_title('Clustering Score with 95% Credible Interval\n(Ground Truth / WSBM / Random baselines)', fontsize=13)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('wsbm_credible_interval.png', dpi=150)
plt.show()
print("Saved → wsbm_credible_interval.png")
