import gc
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from scipy.sparse import load_npz
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args
from torch import nn
from tqdm import tqdm

EPS = 1e-8

space = [
    Integer(800, 1400, name='k'),
    Integer(100, 300, name='d'),
    Real(1e-5, 1e-1, prior='log-uniform', name='learning_rate'),
    Categorical(['Adam', 'AdamW', 'SGD'], name='optimizer'),
]

job_id = os.environ.get("SLURM_JOB_ID", "local")

def log_(x):
    return torch.log(torch.clamp(x, min=EPS))

def dot(x, y):
    return torch.mm(x, y.t())

def nb_dispersion(raw_dispersion, eps=1e-4):
    """Global dispersion parameter r > 0."""
    return F.softplus(raw_dispersion) + eps

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

def wsbm_objective_dense(
    CC,
    q_probs_batch_i,
    q_probs_batch_j,
    U_left,
    U_right,
    bias_exist,
    bias_weight,
    raw_dispersion,
):
    base_scores = dot(U_left, U_right)
    
    #  Bernoulli 
    logits_exist = base_scores + bias_exist
    log_prob_edge = F.logsigmoid(logits_exist)
    log_prob_no_edge = F.logsigmoid(-logits_exist)

    CC_binary = (CC > 0).float()
    
    count_edges_KK = q_probs_batch_i.t() @ CC_binary @ q_probs_batch_j
    mass_i = q_probs_batch_i.sum(dim=0)
    mass_j = q_probs_batch_j.sum(dim=0)
    pair_mass_KK = mass_i.unsqueeze(1) * mass_j.unsqueeze(0)
    count_no_edges_KK = pair_mass_KK - count_edges_KK

    bernoulli_term = (count_edges_KK * log_prob_edge + count_no_edges_KK * log_prob_no_edge).sum()

    # NB
    mu = F.softplus(base_scores + bias_weight) + EPS
    r = nb_dispersion(raw_dispersion)

    count_weighted_KK = q_probs_batch_i.t() @ CC @ q_probs_batch_j

    nb_term = (
        count_weighted_KK * log_(mu)
        - count_weighted_KK * log_(r + mu)
        - count_edges_KK * r * log_(r + mu)  
        + count_edges_KK * r * log_(r)
    ).sum()

    valid_weights = CC[CC > 0]
    gamma_term = (torch.lgamma(valid_weights + r) - torch.lgamma(r)).sum()

    return bernoulli_term + nb_term + gamma_term


def e_step_val(q_logits, val_idx, U_left, U_right, bias_exist, bias_weight, raw_dispersion, val_adj,
               dtype=torch.float32, device='cpu', lr=1e-2, n_iters=10):
    q_val = nn.Parameter(q_logits[val_idx].clone().to(device))
    opt = torch.optim.Adam([q_val], lr=lr)
    CC = torch.tensor(val_adj.toarray(), dtype=dtype, device=device)

    for _ in range(n_iters):
        q_prob = torch.softmax(q_val, dim=-1)
        obj = wsbm_objective_dense(CC, q_prob, q_prob, U_left, U_right, bias_exist, bias_weight, raw_dispersion)
        entropy = -(q_prob * log_(q_prob)).sum(1).mean()
        loss = -(obj + entropy)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        q_logits[val_idx] = q_val.data.to(q_logits.device)

    del CC, q_val, opt


def compute_val_lowerbound(val_idx, q_logits, U_left, U_right, bias_exist, bias_weight, raw_dispersion, val_adj,
                           dtype=torch.float32, device='cpu'):
    with torch.no_grad():
        q = torch.softmax(q_logits[val_idx], dim=-1)
        CC = torch.tensor(val_adj.toarray(), dtype=dtype, device=device)
        obj = wsbm_objective_dense(CC, q, q, U_left, U_right, bias_exist, bias_weight, raw_dispersion)
        entropy = -(q * log_(q)).sum(1).mean()
        loss = -(obj + entropy)

    del CC, q
    return float(loss.item())


def m_step(train_idx=None, train_id_to_local=None, n_max_updates=None, optimizer=None,
           minibatch_size=2500, lr=0.01, q_logits=None, U_left=None, U_right=None,
           bias_exist=None, bias_weight=None, raw_dispersion=None, adj_matrix=None, dtype=torch.float32, device='cpu'):
           
    n_train = len(train_idx)
    train_idx_t = torch.as_tensor(train_idx, device=device, dtype=torch.long)
    initial_loss = 0.0
    final_loss = 0.0

    perm_i = train_idx_t[torch.randperm(n_train, device=device)]
    perm_j = train_idx_t[torch.randperm(n_train, device=device)]
    n_minibatches = max(1, int(np.ceil(n_train / minibatch_size)))

    batches = []
    for i in range(n_minibatches):
        start = i * minibatch_size
        end = min((i + 1) * minibatch_size, n_train)
        if start < end:
            batches.append((perm_i[start:end], perm_j[start:end]))

    def fetch_data(batch):
        mb_i, mb_j = batch
        local_i = mb_i.cpu().numpy().astype(np.int64)
        local_j = mb_j.cpu().numpy().astype(np.int64)
        row_idx = np.array([train_id_to_local[int(x)] for x in local_i], dtype=np.int64)
        col_idx = np.array([train_id_to_local[int(x)] for x in local_j], dtype=np.int64)
        return mb_i, mb_j, adj_matrix[row_idx][:, col_idx].toarray()

    executor = ThreadPoolExecutor(max_workers=4) 
    batch_iterator = executor.map(fetch_data, batches)

    for i, (minibatch_global_i, minibatch_global_j, dense_arr) in enumerate(tqdm(batch_iterator, total=len(batches), desc="M-Step Minibatches", leave=False)):
        if i >= (n_max_updates or 1):
            break

        CC = torch.tensor(dense_arr, dtype=dtype, device=device)

        q_probs_batch_i = torch.softmax(q_logits[minibatch_global_i], dim=-1)
        q_probs_batch_j = torch.softmax(q_logits[minibatch_global_j], dim=-1)

        obj = wsbm_objective_dense(
            CC,
            q_probs_batch_i,
            q_probs_batch_j,
            U_left,
            U_right,
            bias_exist,
            bias_weight,
            raw_dispersion,
        )
        entropy_i = -(q_probs_batch_i * log_(q_probs_batch_i)).sum(1).mean()
        entropy_j = -(q_probs_batch_j * log_(q_probs_batch_j)).sum(1).mean()
        obj = obj + 0.5 * entropy_i + 0.5 * entropy_j

        loss = -obj
        optimizer.zero_grad()
        loss.backward()

        if i == 0:
            initial_loss = float(loss.item())
        final_loss = float(loss.item())

        torch.nn.utils.clip_grad_norm_([U_left, U_right, bias_exist, bias_weight, raw_dispersion, q_logits], 1.0)
        optimizer.step()

        del CC, q_probs_batch_i, q_probs_batch_j, obj, loss

    executor.shutdown()
    
    print(
        f'Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}, '
        f'dispersion r: {nb_dispersion(raw_dispersion).item():.4f}'
    )
    return initial_loss, final_loss


def plot_history(history, plot_dir):
    plt.figure(figsize=(8, 5))
    skip_idx = 1 if len(history['epoch']) > 1 else 0
    
    plt.plot(history['epoch'][skip_idx:], history['train_elbo_end'][skip_idx:], label='Train ELBO (end)')
    plt.plot(history['epoch'][skip_idx:], history['val_elbo'][skip_idx:], label='Validation ELBO')
    plt.plot(history['epoch'][skip_idx:], history['best_val_elbo'][skip_idx:], label='Best Validation ELBO so far')
    
    plt.xlabel('Epoch')
    plt.ylabel('ELBO')
    plt.title('Negative Binomial ELBO vs Epoch (First Epoch Skipped)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / 'elbo_curve.png', dpi=200)
    plt.close()

    np.save(plot_dir / 'epochs.npy', np.array(history['epoch']))
    np.save(plot_dir / 'train_elbo_start.npy', np.array(history['train_elbo_start']))
    np.save(plot_dir / 'train_elbo_end.npy', np.array(history['train_elbo_end']))
    np.save(plot_dir / 'val_elbo.npy', np.array(history['val_elbo']))
    np.save(plot_dir / 'best_val_elbo.npy', np.array(history['best_val_elbo']))

    with open(plot_dir / 'elbo_history.json', 'w') as f:
        json.dump(history, f, indent=2)


def train_once(params, make_plot=False, plot_dir=None, wandb_project='flywriteSBM6_negative_binomial'):
    run = wandb.init(
        project=wandb_project,
        name=f"job_{os.environ.get('SLURM_JOB_ID', 'local')}",
        resume='allow',
        reinit=True,
        settings=wandb.Settings(start_method='thread'),
        config=params,
    )

    K = params['k']
    d = params['d']
    lr = params['learning_rate']
    optimizer_choice = params['optimizer']
    n_epochs = 1000 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(42)
    np.random.seed(42)

    minibatch_size = 5000  
    num_m_updates = 10_000

    dtype = torch.float32

    adj_matrix = load_npz('./sparse_connectivity_matrix_count.npz').astype(np.float32)
    train_idx, val_idx, train_adj, val_adj = train_val_split(adj_matrix, val_frac=0.2)
    train_id_to_local = build_train_id_to_local(train_idx)
    print(f'Loaded count matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.')

    U_left = nn.Parameter(1 / np.sqrt(K * d) * torch.randn(K, d, dtype=dtype, device=device))
    U_right = nn.Parameter(1 / np.sqrt(K * d) * torch.randn(K, d, dtype=dtype, device=device))

    # 根据稀疏度分别初始化两个 bias
    total_elements = adj_matrix.shape[0] * adj_matrix.shape[1]
    density = adj_matrix.nnz / total_elements
    b_exist_init = np.log(density / (1 - density + 1e-6))
    bias_exist = nn.Parameter(torch.tensor([b_exist_init], dtype=dtype, device=device))

    mean_count = float(adj_matrix.data.mean()) if adj_matrix.nnz > 0 else 1.0
    b_weight_init = np.log(np.expm1(max(mean_count, 1e-3)))
    bias_weight = nn.Parameter(torch.tensor([b_weight_init], dtype=dtype, device=device))

    raw_dispersion = nn.Parameter(torch.tensor([1.0], dtype=dtype, device=device))
    q_logits = nn.Parameter(1 / K * torch.randn(adj_matrix.shape[0], K, dtype=dtype, device=device))

    params_to_optimize = [U_left, U_right, bias_exist, bias_weight, raw_dispersion, q_logits]
    if optimizer_choice == 'Adam':
        optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    elif optimizer_choice == 'SGD':
        optimizer = torch.optim.SGD(params_to_optimize, lr=lr)
    else:
        optimizer = torch.optim.AdamW(params_to_optimize, lr=lr)

    lowest_elbo = float('inf')
    best_epoch = 0
    history = {
        'epoch': [],
        'train_elbo_start': [],
        'train_elbo_end': [],
        'val_elbo': [],
        'best_val_elbo': [],
        'dispersion_r': [],
    }

    del adj_matrix

    if make_plot:
        if plot_dir is None:
            run_name = run.id if run is not None else 'offline_run'
            plot_dir = Path('elbo_plots_wsbm') / f'best_run_{run_name}'
        else:
            plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

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
                bias_exist=bias_exist,
                bias_weight=bias_weight,
                raw_dispersion=raw_dispersion,
                adj_matrix=train_adj,
                dtype=dtype,
                n_max_updates=num_m_updates,
                device=device,
            )

            print('\n--- Validation Step ---')
            e_step_val(q_logits, val_idx, U_left, U_right, bias_exist, bias_weight, raw_dispersion, val_adj, device=device, dtype=dtype)

            val_loss = compute_val_lowerbound(
                val_idx=val_idx,
                q_logits=q_logits,
                U_left=U_left,
                U_right=U_right,
                bias_exist=bias_exist,
                bias_weight=bias_weight,
                raw_dispersion=raw_dispersion,
                val_adj=val_adj,
                dtype=dtype,
                device=device,
            )
            train_elbo_start = -initial_loss
            train_elbo_end = -final_loss
            current_val_elbo = -val_loss
            current_r = float(nb_dispersion(raw_dispersion).detach().cpu().item())

            if val_loss < lowest_elbo:
                lowest_elbo = val_loss
                best_epoch = epoch

            history['epoch'].append(epoch + 1)
            history['train_elbo_start'].append(train_elbo_start)
            history['train_elbo_end'].append(train_elbo_end)
            history['val_elbo'].append(current_val_elbo)
            history['best_val_elbo'].append(-lowest_elbo)
            history['dispersion_r'].append(current_r)

            print(f'Current Validation ELBO: {current_val_elbo}')
            print(f'Best Validation ELBO so far: {-lowest_elbo}')
            print(f'Current dispersion r: {current_r}')

            run.log({
                'train_elbo_start': train_elbo_start,
                'train_elbo_end': train_elbo_end,
                'val_elbo': current_val_elbo,
                'best_val_elbo': -lowest_elbo,
                'dispersion_r': current_r,
                'epoch': epoch + 1,
            }, commit=True)

            if make_plot:
                plot_history(history, plot_dir)

    except KeyboardInterrupt:
        print('Training interrupted.')

    print(f'Best Epoch: {best_epoch + 1}')
    print(f'Best Validation ELBO: {-lowest_elbo}')
    print(f'Final dispersion r: {float(nb_dispersion(raw_dispersion).detach().cpu().item())}')

    if make_plot:
        safe_params = {key: (val.item() if hasattr(val, 'item') else val) for key, val in params.items()}
        with open(plot_dir / 'best_run_summary.json', 'w') as f:
            json.dump({
                'best_epoch': best_epoch + 1,
                'best_val_elbo': -lowest_elbo,
                'final_dispersion_r': float(nb_dispersion(raw_dispersion).detach().cpu().item()),
                'params': safe_params,
            }, f, indent=2)

    del optimizer, train_adj, val_adj
    run.finish()
    return lowest_elbo, best_epoch, history


@use_named_args(space)
def objective(**params):
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    trial_id = objective.call_count
    objective.call_count += 1

    plot_dir = Path('elbo_plots_wsbm') / f'trial_{job_id}_{trial_id}'
    plot_dir.mkdir(parents=True, exist_ok=True)

    lowest_elbo, _, _ = train_once(
        params,
        make_plot=True,
        plot_dir=plot_dir,
        wandb_project='flywriteSBM6_negative_binomial_trials',
    )
    return lowest_elbo

objective.call_count = 0


def save_best_results(res_gp, output_path=None):
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    if output_path is None:
        output_path = f"results/best_hyperparams_{job_id}.txt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        print('Best Validation ELBO: {:.6f}'.format(res_gp.fun), file=f)
        print('Best hyperparameters:', file=f)
        print('  k:', res_gp.x[0], file=f)
        print('  d:', res_gp.x[1], file=f)
        print('  learning_rate:', res_gp.x[2], file=f)
        print('  optimizer:', res_gp.x[3], file=f)
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
    print('  optimizer:', res_gp.x[3])

    best_params = {
        'k': res_gp.x[0],
        'd': res_gp.x[1],
        'learning_rate': res_gp.x[2],
        'optimizer': res_gp.x[3],
    }

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    best_plot_dir = Path('elbo_plots_wsbm') / f'best_run_{job_id}'
    best_plot_dir.mkdir(parents=True, exist_ok=True)

    with open(best_plot_dir / 'best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    print('\nRe-running best hyperparameters to generate the 1000-epoch ELBO curve...')
    best_loss, best_epoch, _ = train_once(
        best_params,
        make_plot=True,
        plot_dir=best_plot_dir,
        wandb_project='flywriteSBM6_negative_binomial_best_run',
    )

    print(f'Best-run validation ELBO: {-best_loss}')
    print(f'Best-run best epoch: {best_epoch + 1}')
    print(f'ELBO curve saved to: {best_plot_dir / "elbo_curve.png"}')