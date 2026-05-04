import os
import gc
import random
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.sparse import load_npz
from tqdm import tqdm


EPS = 1e-8


def log_(x):
    return torch.log(torch.clamp(x, min=EPS))


def dot(x, y):
    return torch.mm(x, y.t())


def nb_dispersion(raw_dispersion, eps=1e-4):
    """Global dispersion parameter r > 0."""
    return F.softplus(raw_dispersion) + eps


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
    """
    Hurdle WSBM (Bernoulli + Negative Binomial)
    """
    base_scores = dot(U_left, U_right)
    
    # === 1. Bernoulli 部分 (拓扑结构) ===
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

    # === 2. NB 部分 (仅作用于实际存在的边) ===
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


# 优化：将数据加载提取到循环外部，避免重复读取导致 I/O 阻塞
adj_matrix = load_npz('./sparse_connectivity_matrix_count.npz').astype(np.float32)
N = adj_matrix.shape[0]
print(f'Loaded count matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.')

for task_id in range(0, 10):
    print(f"\n========== Starting Task {task_id} ==========")
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'

    torch.manual_seed(random.randint(1, 1000))

    K = 1086
    d = 138
    minibatch_size = 5000
    num_m_updates = 10_000
    n_epochs = 1000

    dtype = torch.float32

    U_left = nn.Parameter(1 / np.sqrt(K * d) * torch.randn(K, d, dtype=dtype).to(device))
    U_right = nn.Parameter(1 / np.sqrt(K * d) * torch.randn(K, d, dtype=dtype).to(device))

    # 根据稀疏度分别初始化两个 bias
    total_elements = N * N
    density = adj_matrix.nnz / total_elements
    b_exist_init = np.log(density / (1 - density + 1e-6))
    bias_exist = nn.Parameter(torch.tensor([b_exist_init], dtype=dtype, device=device))

    mean_count = float(adj_matrix.data.mean()) if adj_matrix.nnz > 0 else 1.0
    b_weight_init = np.log(np.expm1(max(mean_count, 1e-3)))
    bias_weight = nn.Parameter(torch.tensor([b_weight_init], dtype=dtype, device=device))

    raw_dispersion = nn.Parameter(torch.tensor([1.0], dtype=dtype, device=device))
    q_logits = nn.Parameter(1 / K * torch.randn(N, K, dtype=dtype).to(device))

    optimizer = torch.optim.AdamW([U_left, U_right, bias_exist, bias_weight, raw_dispersion, q_logits], lr=0.01)

    def m_step(n_max_updates=None, optimizer=None):
        if n_max_updates is None:
            n_max_updates = 1

        initial_loss = 0.0
        final_loss = 0.0

        indices_i = torch.randperm(N, device='cpu')
        indices_j = torch.randperm(N, device='cpu')
        n_minibatches = max(1, int(np.ceil(N / minibatch_size)))

        # 准备批次索引
        batches = []
        for i in range(n_minibatches):
            start = i * minibatch_size
            end = min((i + 1) * minibatch_size, N)
            if start < end:
                batches.append((indices_i[start:end].numpy(), indices_j[start:end].numpy()))

        # 异步多线程 CPU 提取稀疏矩阵，避免 GPU 饥饿
        def fetch_data(batch):
            row_ids_np, col_ids_np = batch
            return row_ids_np, col_ids_np, adj_matrix[row_ids_np][:, col_ids_np].toarray()

        executor = ThreadPoolExecutor(max_workers=4)
        batch_iterator = executor.map(fetch_data, batches)

        for i, (row_ids_np, col_ids_np, dense_arr) in enumerate(tqdm(batch_iterator, total=len(batches), desc="Minibatches", leave=False)):
            if i >= n_max_updates:
                print(f'Reached maximum number of updates {n_max_updates}.')
                break

            row_ids_gpu = torch.from_numpy(row_ids_np).to(device)
            col_ids_gpu = torch.from_numpy(col_ids_np).to(device)

            q_probs_batch_i = torch.softmax(q_logits[row_ids_gpu], dim=-1)
            q_probs_batch_j = torch.softmax(q_logits[col_ids_gpu], dim=-1)

            CC = torch.tensor(dense_arr, dtype=dtype, device=device)

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
                initial_loss = loss.item()
            final_loss = loss.item()

            torch.nn.utils.clip_grad_norm_([U_left, U_right, bias_exist, bias_weight, raw_dispersion, q_logits], 1.0)
            optimizer.step()

            del CC, q_probs_batch_i, q_probs_batch_j, obj, loss, entropy_i, entropy_j, row_ids_gpu, col_ids_gpu

        executor.shutdown()
        print(f'Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}, dispersion r: {nb_dispersion(raw_dispersion).item():.4f}')

    try:
        for epoch in range(n_epochs):
            print(f'Epoch {epoch}')
            m_step(n_max_updates=num_m_updates, optimizer=optimizer)

    except KeyboardInterrupt:
        print('Training interrupted.')

    cluster_assignments = torch.argmax(q_logits, dim=-1).cpu().numpy()
    cluster_scores = torch.max(q_logits, dim=-1).values.cpu().detach().numpy()
    U_left_final = U_left.detach().cpu().numpy()
    U_right_final = U_right.detach().cpu().numpy()
    
    # 获取分离的两个 bias 参数
    bias_exist_final = bias_exist.detach().cpu().numpy() 
    bias_weight_final = bias_weight.detach().cpu().numpy() 
    dispersion_final = nb_dispersion(raw_dispersion).detach().cpu().numpy()

    mapping_path = './root_id_to_index_mapping_count.json'
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    idx_to_location = {int(v): int(k) for k, v in mapping.items()}

    cluster_assignment_dict = {}
    for i in range(len(cluster_assignments)):
        location_id = idx_to_location.get(i, i)
        cluster_assignment_dict[location_id] = int(cluster_assignments[i])

    folder_name = f'credible_interval_results_flywire_wsb/sbm_{task_id}'
    os.makedirs(folder_name, exist_ok=True)

    np.save(os.path.join(folder_name, 'cluster_assignments.npy'), cluster_assignments)
    np.save(os.path.join(folder_name, 'cluster_scores.npy'), cluster_scores)
    np.save(os.path.join(folder_name, 'U_left.npy'), U_left_final)
    np.save(os.path.join(folder_name, 'U_right.npy'), U_right_final)
    
    with open(os.path.join(folder_name, 'cluster_assignment_dict.json'), 'w') as f:
        json.dump(cluster_assignment_dict, f)
        
    # 分别保存拆分的 bias
    np.save(os.path.join(folder_name, 'bias_exist.npy'), bias_exist_final)
    np.save(os.path.join(folder_name, 'bias_weight.npy'), bias_weight_final)
    np.save(os.path.join(folder_name, 'dispersion.npy'), dispersion_final)

    print("Results saved: 'cluster_assignments.npy', 'U_left.npy', 'U_right.npy', 'cluster_assignment_dict.json', 'bias_exist.npy', 'bias_weight.npy', 'dispersion.npy'")

    del U_left, U_right, bias_exist, bias_weight, raw_dispersion, q_logits, optimizer
    del cluster_assignments, cluster_scores, U_left_final, U_right_final, bias_exist_final, bias_weight_final, dispersion_final
    gc.collect()
    
    if device == 'cuda':
        torch.cuda.empty_cache()