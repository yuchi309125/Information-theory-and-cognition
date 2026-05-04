import gc
import os

import numpy as np
import torch
from scipy.sparse import load_npz
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import wandb

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args


EPS = 1e-8

space = [
    Integer(900, 1200, name='k'),
    Integer(100, 200, name='d'),
    Real(1e-5, 1e-1, prior='log-uniform', name='learning_rate'),
    Integer(10, 11, name='n_epochs'),
    Categorical(['Adam', 'AdamW', 'SGD'], name='optimizer'),
]


def log_(x):
    return torch.log(torch.clamp(x, min=EPS))


def dot(x, y):
    return torch.mm(x, y.t())


def poisson_mean(U_left, U_right, bias, eps=1e-6):
    """Mean parameter mu_{kk'} = softplus(u_k^T v_{k'} + b)."""
    return F.softplus(dot(U_left, U_right) + bias) + eps


def train_val_split(adj_matrix, val_frac=0.2, seed=42):
    n = adj_matrix.shape[0]
    np.random.seed(seed)
    perm = np.random.permutation(n)
    n_val = int(n * val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_adj = adj_matrix[train_idx][:, train_idx]
    val_adj = adj_matrix[val_idx][:, val_idx]
    return train_idx, val_idx, train_adj, val_adj


def build_train_id_to_local(train_idx):
    return {int(node_id): local for local, node_id in enumerate(train_idx)}


def poisson_objective_dense(CC, q_probs_batch_i, q_probs_batch_j, U_left, U_right, bias):
    """
    ELBO contribution up to a constant in Y for a dense minibatch CC.

    For the Poisson likelihood,
        log p(y | mu) = y log mu - mu - log(y!)
    so the only removable term is the pure data constant -log(y!).
    """
    mu = poisson_mean(U_left, U_right, bias).to(dtype=CC.dtype, device=CC.device)

    count_weighted_KK = q_probs_batch_i.t() @ CC @ q_probs_batch_j
    mass_i = q_probs_batch_i.sum(dim=0)
    mass_j = q_probs_batch_j.sum(dim=0)
    pair_mass_KK = mass_i.unsqueeze(1) * mass_j.unsqueeze(0)

    return (count_weighted_KK * log_(mu) - pair_mass_KK * mu).sum()


def e_step_val(q_logits, val_idx, U_left, U_right, bias, val_adj,
               dtype=torch.float32, device='cpu', lr=1e-2, n_iters=10):
    q_val = nn.Parameter(q_logits[val_idx].clone().to(device))
    opt = torch.optim.Adam([q_val], lr=lr)
    CC = torch.tensor(val_adj.toarray(), dtype=dtype, device=device)

    for _ in range(n_iters):
        q_prob = torch.softmax(q_val, dim=-1)
        obj = poisson_objective_dense(CC, q_prob, q_prob, U_left, U_right, bias)
        entropy = -(q_prob * log_(q_prob)).sum(1).mean()
        loss = -(obj + entropy)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        q_logits[val_idx] = q_val.data.to(q_logits.device)

    del CC, q_val, opt
    gc.collect()


def compute_val_lowerbound(val_idx, q_logits, U_left, U_right, bias, val_adj,
                           dtype=torch.float32, device='cpu'):
    with torch.no_grad():
        q = torch.softmax(q_logits[val_idx], dim=-1)
        CC = torch.tensor(val_adj.toarray(), dtype=dtype, device=device)
        obj = poisson_objective_dense(CC, q, q, U_left, U_right, bias)
        entropy = -(q * log_(q)).sum(1).mean()
        loss = -(obj + entropy)

    del CC, q, obj, entropy
    gc.collect()
    return float(loss.item())


def m_step(train_idx=None, train_id_to_local=None, n_max_updates=None, optimizer=None,
           minibatch_size=2500, lr=0.01, q_logits=None, U_left=None, U_right=None,
           bias=None, adj_matrix=None, dtype=torch.float32, device='cpu'):
    if q_logits is None or U_left is None or U_right is None or bias is None or adj_matrix is None:
        return 0.0, 0.0
    if train_idx is None or train_id_to_local is None:
        raise ValueError('train_idx and train_id_to_local must be provided.')
    if n_max_updates is None:
        n_max_updates = 1

    if optimizer is None:
        optimizer = torch.optim.AdamW([U_left, U_right, bias, q_logits], lr=lr)

    n_train = len(train_idx)
    train_idx_t = torch.as_tensor(train_idx, device=device, dtype=torch.long)
    initial_loss = 0.0
    final_loss = 0.0

    perm_i = train_idx_t[torch.randperm(n_train, device=device)]
    perm_j = train_idx_t[torch.randperm(n_train, device=device)]
    n_minibatches = max(1, int(np.ceil(n_train / minibatch_size)))

    for i in tqdm(range(n_minibatches)):
        if i >= n_max_updates:
            print(f'Reached maximum number of updates {n_max_updates}.')
            break

        start = i * minibatch_size
        end = min((i + 1) * minibatch_size, n_train)
        minibatch_global_i = perm_i[start:end]
        minibatch_global_j = perm_j[start:end]

        if len(minibatch_global_i) == 0 or len(minibatch_global_j) == 0:
            continue

        q_probs_batch_i = torch.softmax(q_logits[minibatch_global_i], dim=-1)
        q_probs_batch_j = torch.softmax(q_logits[minibatch_global_j], dim=-1)

        local_i = minibatch_global_i.detach().cpu().numpy().astype(np.int64)
        local_j = minibatch_global_j.detach().cpu().numpy().astype(np.int64)
        row_idx = np.array([train_id_to_local[int(x)] for x in local_i], dtype=np.int64)
        col_idx = np.array([train_id_to_local[int(x)] for x in local_j], dtype=np.int64)

        CC = torch.tensor(
            adj_matrix[row_idx][:, col_idx].toarray(),
            dtype=dtype,
            device=device,
        )

        obj = poisson_objective_dense(CC, q_probs_batch_i, q_probs_batch_j, U_left, U_right, bias)
        entropy_i = -(q_probs_batch_i * log_(q_probs_batch_i)).sum(1).mean()
        entropy_j = -(q_probs_batch_j * log_(q_probs_batch_j)).sum(1).mean()
        obj = obj + 0.5 * entropy_i + 0.5 * entropy_j

        loss = -obj
        optimizer.zero_grad()
        loss.backward()

        if i == 0:
            initial_loss = float(loss.item())
        final_loss = float(loss.item())

        torch.nn.utils.clip_grad_norm_([U_left, U_right, bias, q_logits], 1.0)
        optimizer.step()

        del CC, q_probs_batch_i, q_probs_batch_j, obj, loss, entropy_i, entropy_j
        gc.collect()

    print(f'Initial loss: {initial_loss}, Final loss: {final_loss}')
    gc.collect()
    return initial_loss, final_loss


@use_named_args(space)
def objective(**params):
    run = wandb.init(
        project='flywriteSBM6_poisson',
        resume='allow',
        reinit=True,
        settings=wandb.Settings(start_method='thread'),
        config=params,
    )

    K = params['k']
    d = params['d']
    lr = params['learning_rate']
    n_epochs = params['n_epochs']
    optimizer_choice = params['optimizer']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(42)
    np.random.seed(42)

    minibatch_size = 10_000
    num_m_updates = 10_000

    dtype = torch.float32
    if device == 'cuda':
        dtype = torch.float32

    adj_matrix = load_npz('./sparse_connectivity_matrix_1.npz').astype(np.float32)
    train_idx, val_idx, train_adj, val_adj = train_val_split(adj_matrix, val_frac=0.2)
    train_id_to_local = build_train_id_to_local(train_idx)
    print(f'Loaded count matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.')

    U_left = nn.Parameter(1 / np.sqrt(K * d) * torch.randn(K, d, dtype=dtype, device=device))
    U_right = nn.Parameter(1 / np.sqrt(K * d) * torch.randn(K, d, dtype=dtype, device=device))

    mean_count = float(adj_matrix.data.mean()) if adj_matrix.nnz > 0 else 1.0
    bias_init = np.log(np.expm1(max(mean_count, 1e-3)))
    bias = nn.Parameter(torch.tensor([bias_init], dtype=dtype, device=device))
    q_logits = nn.Parameter(1 / K * torch.randn(adj_matrix.shape[0], K, dtype=dtype, device=device))

    params_to_optimize = [U_left, U_right, bias, q_logits]
    if optimizer_choice == 'Adam':
        optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    elif optimizer_choice == 'SGD':
        optimizer = torch.optim.SGD(params_to_optimize, lr=lr)
    else:
        optimizer = torch.optim.AdamW(params_to_optimize, lr=lr)

    lowest_elbo = float('inf')
    best_epoch = 0

    del adj_matrix
    gc.collect()

    try:
        for epoch in range(n_epochs):
            print(f'\nEpoch {epoch + 1}/{n_epochs}')
            initial_loss, final_loss = m_step(
                train_idx=train_idx,
                train_id_to_local=train_id_to_local,
                optimizer=optimizer,
                minibatch_size=minibatch_size,
                lr=lr,
                q_logits=q_logits,
                U_left=U_left,
                U_right=U_right,
                bias=bias,
                adj_matrix=train_adj,
                dtype=dtype,
                n_max_updates=num_m_updates,
                device=device,
            )

            print('\n--- Validation Step ---')
            e_step_val(q_logits, val_idx, U_left, U_right, bias, val_adj, device=device, dtype=dtype)

            val_elbo = compute_val_lowerbound(
                val_idx=val_idx,
                q_logits=q_logits,
                U_left=U_left,
                U_right=U_right,
                bias=bias,
                val_adj=val_adj,
                dtype=dtype,
                device=device,
            )
            print(f'Current Validation ELBO: {val_elbo}')

            if val_elbo < lowest_elbo:
                lowest_elbo = val_elbo
                best_epoch = epoch

            run.log({
                'initial_train_elbo': initial_loss,
                'final_train_elbo': final_loss,
                'val_elbo': val_elbo,
                'best_val_elbo': lowest_elbo,
                'epoch': epoch,
            }, commit=True)

    except KeyboardInterrupt:
        print('Training interrupted.')

    print(f'Best Epoch: {best_epoch}')
    print(f'Best Validation ELBO: {lowest_elbo}')

    del optimizer, dtype, num_m_updates, minibatch_size, d, train_adj, val_adj
    gc.collect()
    run.finish()
    return lowest_elbo


def save_best_results(res_gp, output_path='best_hyperparams_results.txt'):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        print('Best Validation ELBO: {:.6f}'.format(res_gp.fun), file=f)
        print('Best hyperparameters:', file=f)
        print('  k:', res_gp.x[0], file=f)
        print('  d:', res_gp.x[1], file=f)
        print('  learning_rate:', res_gp.x[2], file=f)
        print('  n_epochs:', res_gp.x[3], file=f)
        print('  optimizer:', res_gp.x[4], file=f)
        print('\nBest params:', res_gp.x, file=f)



if __name__ == '__main__':
    res_gp = gp_minimize(
        objective,
        space,
        acq_func='EI',
        n_calls=10,
        random_state=42,
    )

    save_best_results(res_gp)
    print('Best Validation ELBO:', res_gp.fun)
    print('Best hyperparameters:')
    print('  k:', res_gp.x[0])
    print('  d:', res_gp.x[1])
    print('  learning_rate:', res_gp.x[2])
    print('  n_epochs:', res_gp.x[3])
    print('  optimizer:', res_gp.x[4])
