import torch
from torch import nn
import torch.nn.functional as F
from scipy.sparse import load_npz
import numpy as np
from tqdm import tqdm
import gc
import wandb
import sys
import json


def log_(x):
    return torch.log(torch.clamp(x, min=1e-8))


def dot(x, y):
    return torch.mm(x, y.t())


def nb_mean(U_left, U_right, bias, eps=1e-6):
    return F.softplus(dot(U_left, U_right) + bias) + eps


def train_val_split(adj_matrix, val_frac=0.2, seed=42):
    N = adj_matrix.shape[0]
    np.random.seed(seed)
    perm = np.random.permutation(N)
    n_val = int(N * val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_adj = adj_matrix[train_idx][:, train_idx]
    val_adj = adj_matrix[val_idx][:, val_idx]
    return train_idx, val_idx, train_adj, val_adj


def nb_objective_dense(CC, q_probs_batch_i, q_probs_batch_j, U_left, U_right, bias, log_theta):
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


def e_step_val(q_logits, val_idx, U_left, U_right, bias, log_theta, val_adj,
               dtype=torch.float32, device='cpu', lr=1e-2, n_iters=10):
    q_val = nn.Parameter(q_logits[val_idx].clone().to(device))
    opt = torch.optim.Adam([q_val], lr=lr)
    CC = torch.tensor(val_adj.toarray(), dtype=dtype, device=device)

    for _ in range(n_iters):
        q_prob = torch.softmax(q_val, dim=-1)
        obj = nb_objective_dense(CC, q_prob, q_prob, U_left, U_right, bias, log_theta)
        obj = obj - (q_prob * log_(q_prob)).sum(1).mean()
        loss = -obj
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        q_logits[val_idx] = q_val.data.cpu()

    del CC, q_val, opt
    gc.collect()


def compute_val_lowerbound(val_idx, q_logits, U_left, U_right, bias, log_theta, val_adj,
                           dtype=torch.float32, device='cpu'):
    with torch.no_grad():
        q = torch.softmax(q_logits[val_idx], dim=-1)
        CC = torch.tensor(val_adj.toarray(), dtype=dtype, device=device)
        obj = nb_objective_dense(CC, q, q, U_left, U_right, bias, log_theta)
        entropy = (q * log_(q)).sum(1).mean()
        loss = -(obj - entropy)

    del CC, q, obj, entropy
    gc.collect()
    return loss.item()


def m_step(N=None, n_max_updates=None, optimizer=None, minibatch_size=2500, lr=0.01,
           q_logits=None, U_left=None, U_right=None, bias=None, log_theta=None,
           adj_matrix=None, dtype=torch.float32, device='cpu'):
    if q_logits is None or U_left is None or U_right is None or bias is None or log_theta is None or adj_matrix is None:
        return
    if N is None:
        N = 1
    if n_max_updates is None:
        n_max_updates = 1

    if optimizer is None:
        optimizer = torch.optim.AdamW([U_left, U_right, bias, log_theta, q_logits], lr=lr)

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

        obj = nb_objective_dense(CC, q_probs_batch_i, q_probs_batch_j, U_left, U_right, bias, log_theta)
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
    del U_left, U_right, bias, log_theta, q_logits, optimizer, N, adj_matrix, dtype, minibatch_size, lr, indices_i, indices_j, n_minibatches, n_max_updates
    gc.collect()
    return initial_loss, final_loss


def objective(params):
    run = wandb.init(
        project='flywriteSBM6_nb',
        resume='allow',
        reinit=True,
        settings=wandb.Settings(start_method='thread')
    )

    K = params['k']
    d = params['d']
    lr = params['learning_rate']
    n_epochs = 1000
    optimizer_choice = params['optimizer']
    device = 'cpu'

    torch.manual_seed(42)

    minibatch_size = 10_000
    num_m_updates = 10_000

    dtype = torch.float32
    if device == 'cuda':
        dtype = torch.bfloat16

    adj_matrix = load_npz('./sparse_connectivity_matrix.npz').astype(np.float32)
    N = adj_matrix.shape[0]
    train_idx, val_idx, train_adj, val_adj = train_val_split(adj_matrix, val_frac=0.2)
    N_train = train_idx.shape[0]
    print(f'Loaded count matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.')

    U_left = nn.Parameter(1 / np.sqrt(K * d) * torch.randn(K, d, dtype=dtype).to(device))
    U_right = nn.Parameter(1 / np.sqrt(K * d) * torch.randn(K, d, dtype=dtype).to(device))

    mean_count = float(adj_matrix.data.mean()) if adj_matrix.nnz > 0 else 1.0
    bias = nn.Parameter(torch.log(torch.tensor([max(mean_count, 1e-3)], dtype=dtype).to(device)))
    log_theta = nn.Parameter(torch.log(torch.tensor([1.0], dtype=dtype).to(device)))
    q_logits = nn.Parameter(1 / K * torch.randn(N, K, dtype=dtype).to(device))

    if optimizer_choice == 'Adam':
        optimizer = torch.optim.Adam([U_left, U_right, bias, log_theta, q_logits], lr=lr)
    elif optimizer_choice == 'SGD':
        optimizer = torch.optim.SGD([U_left, U_right, bias, log_theta, q_logits], lr=lr)
    else:
        optimizer = torch.optim.AdamW([U_left, U_right, bias, log_theta, q_logits], lr=lr)

    lowest_elbo = float('inf')
    best_epoch = 0

    del adj_matrix
    gc.collect()

    try:
        for epoch in range(n_epochs):
            print(f'\nEpoch {epoch + 1}/{n_epochs}')
            initial_loss, final_loss = m_step(
                N=N_train,
                optimizer=optimizer,
                minibatch_size=minibatch_size,
                lr=lr,
                q_logits=q_logits,
                U_left=U_left,
                U_right=U_right,
                bias=bias,
                log_theta=log_theta,
                adj_matrix=train_adj,
                dtype=dtype,
                n_max_updates=num_m_updates,
                device=device
            )

            print('\n--- Validation Step ---')
            e_step_val(q_logits, val_idx, U_left, U_right, bias, log_theta, val_adj, device=device, dtype=dtype)

            val_elbo = compute_val_lowerbound(
                val_idx=val_idx,
                q_logits=q_logits,
                U_left=U_left,
                U_right=U_right,
                bias=bias,
                log_theta=log_theta,
                val_adj=val_adj,
                dtype=dtype,
                device=device
            )
            print(f'Current Validation ELBO: {val_elbo}')

            if val_elbo < lowest_elbo:
                lowest_elbo = val_elbo
                best_epoch = epoch

            run.log({
                'initial_train_elbo': initial_loss,
                'final_train_elbo': final_loss,
                'val_elbo': val_elbo,
                'theta': float(F.softplus(log_theta).detach().cpu()),
            }, commit=True)

    except KeyboardInterrupt:
        print('Training interrupted.')

    print(f'Best Epoch: {best_epoch}')
    print(f'Best Validation ELBO: {lowest_elbo}')

    del optimizer, dtype, num_m_updates, minibatch_size, d
    gc.collect()
    run.finish()
    return lowest_elbo, best_epoch


task_id = int(sys.argv[1])

with open(f'configs/params_{task_id}.json') as f:
    params = json.load(f)

elbo, best_epoch = objective(params)

with open(f'results/result_{task_id}.json', 'w') as f:
    json.dump({'elbo': elbo, 'best_epoch': best_epoch}, f)
