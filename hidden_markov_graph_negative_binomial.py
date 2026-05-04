import os
import sys
import gc
import random
import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.sparse import load_npz
from tqdm import tqdm

device = 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
if torch.cuda.is_available():
    device = 'cuda'

torch.manual_seed(random.randint(1, 1000))

K = 1859
d = 195
minibatch_size = 10_000
num_m_updates = 10_000
n_epochs = 1_000

dtype = torch.float32
if device == 'cuda':
    dtype = torch.bfloat16


def log_(x):
    return torch.log(torch.clamp(x, min=1e-8))


def dot(x, y):
    return torch.mm(x, y.t())


def nb_mean(U_left, U_right, bias, eps=1e-6):
    return F.softplus(dot(U_left, U_right) + bias) + eps


adj_matrix = load_npz('./sparse_connectivity_matrix.npz').astype(np.float32)
N = adj_matrix.shape[0]
print(f'Loaded count matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.')

U_left = nn.Parameter(1 / np.sqrt(K * d) * torch.randn(K, d, dtype=dtype).to(device))
U_right = nn.Parameter(1 / np.sqrt(K * d) * torch.randn(K, d, dtype=dtype).to(device))

mean_count = float(adj_matrix.data.mean()) if adj_matrix.nnz > 0 else 1.0
bias = nn.Parameter(torch.log(torch.tensor([max(mean_count, 1e-3)], dtype=dtype).to(device)))
log_theta = nn.Parameter(torch.log(torch.tensor([1.0], dtype=dtype).to(device)))
q_logits = nn.Parameter(1 / K * torch.randn(N, K, dtype=dtype).to(device))


def nb_objective_dense(CC, q_probs_batch_i, q_probs_batch_j):
    mu = nb_mean(U_left, U_right, bias)
    theta = F.softplus(log_theta).to(dtype=CC.dtype, device=CC.device) + 1e-6

    log_theta_val = log_(theta)
    log_mu = log_(mu)
    log_theta_plus_mu = log_(theta + mu)

    count_weighted_KK = torch.mm(q_probs_batch_i.t(), torch.mm(CC, q_probs_batch_j))
    mass_i = q_probs_batch_i.sum(0)
    mass_j = q_probs_batch_j.sum(0)
    pair_mass_KK = mass_i.unsqueeze(1) * mass_j.unsqueeze(0)

    block_term = (
        count_weighted_KK * (log_mu - log_theta_plus_mu)
        + pair_mass_KK * theta * (log_theta_val - log_theta_plus_mu)
    ).sum()

    const_theta_term = (
        torch.lgamma(CC + theta)
        - torch.lgamma(theta)
        - torch.lgamma(CC + 1.0)
    ).sum()

    return const_theta_term + block_term


def m_step(n_max_updates=None, optimizer=None):
    if n_max_updates is None:
        n_max_updates = 1
    if optimizer is None:
        optimizer = torch.optim.AdamW([U_left, U_right, bias, log_theta, q_logits], lr=0.001)

    initial_loss = 0.0
    final_loss = 0.0

    indices_i = torch.randperm(N, device=device)
    indices_j = torch.randperm(N, device=device)
    n_minibatches = max(1, N // minibatch_size)

    for i in tqdm(range(n_minibatches)):
        if i > n_max_updates:
            print(f'Reached maximum number of updates {n_max_updates}.')
            break

        minibatch_indices_i = indices_i[i * minibatch_size: min((i + 1) * minibatch_size, N)]
        minibatch_indices_j = indices_j[i * minibatch_size: min((i + 1) * minibatch_size, N)]

        if len(minibatch_indices_i) == 0 or len(minibatch_indices_j) == 0:
            continue

        q_probs_batch_i = torch.softmax(q_logits[minibatch_indices_i], dim=-1)
        q_probs_batch_j = torch.softmax(q_logits[minibatch_indices_j], dim=-1)

        CC = torch.tensor(
            adj_matrix[minibatch_indices_i.cpu().numpy()][:, minibatch_indices_j.cpu().numpy()].toarray(),
            dtype=dtype
        ).to(device)

        obj = nb_objective_dense(CC, q_probs_batch_i, q_probs_batch_j)
        obj = obj - (q_probs_batch_i * log_(q_probs_batch_i)).sum(1).mean() / 2
        obj = obj - (q_probs_batch_j * log_(q_probs_batch_j)).sum(1).mean() / 2

        loss = -obj
        optimizer.zero_grad()
        loss.backward()

        if i == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

        torch.nn.utils.clip_grad_norm_([U_left, U_right, bias, log_theta, q_logits], 1.0)
        optimizer.step()

        del CC, q_probs_batch_i, q_probs_batch_j, minibatch_indices_i, minibatch_indices_j, obj, loss
        gc.collect()

    print(f'Initial loss: {initial_loss}, Final loss: {final_loss}')


optimizer = torch.optim.AdamW([U_left, U_right, bias, log_theta, q_logits], lr=0.00999)

try:
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        m_step(n_max_updates=num_m_updates, optimizer=optimizer)
        print('Cluster assignments:')
        for i in range(min(10, N)):
            print(f'Node {i}: Cluster {torch.argmax(q_logits[i]).item()}')

except KeyboardInterrupt:
    print('Training interrupted.')

cluster_assignments = torch.argmax(q_logits, dim=-1).cpu().numpy()
cluster_scores = torch.max(q_logits, dim=-1).values.cpu().detach().numpy()
U_left_final = U_left.detach().cpu().numpy()
U_right_final = U_right.detach().cpu().numpy()
theta_final = float(F.softplus(log_theta).detach().cpu())

mapping_path = './location_id_to_index_mapping.json'


with open(mapping_path, 'r') as f:
    mapping = json.load(f)

idx_to_location = {int(v): int(k) for k, v in mapping.items()}

cluster_assignment_dict = {}
for i in range(len(cluster_assignments)):
    location_id = idx_to_location[i]
    cluster_assignment_dict[location_id] = int(cluster_assignments[i])

task_id = int(sys.argv[1])

folder_name = f'credible_interval_results/sbm_{task_id}'
os.makedirs(folder_name, exist_ok=True)

np.save(os.path.join(folder_name, 'cluster_assignments.npy'), cluster_assignments)
np.save(os.path.join(folder_name, 'cluster_scores.npy'), cluster_scores)
np.save(os.path.join(folder_name, 'U_left.npy'), U_left_final)
np.save(os.path.join(folder_name, 'U_right.npy'), U_right_final)
np.save(os.path.join(folder_name, 'cluster_assignment_dict.npy'), cluster_assignment_dict)
np.save(os.path.join(folder_name, 'theta.npy'), np.array([theta_final], dtype=np.float32))

print("Results saved: 'cluster_assignments.npy', 'U_left.npy', 'U_right.npy', 'cluster_assignment_dict.npy', 'theta.npy'")
