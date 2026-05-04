import torch
from torch import nn
from scipy.sparse import load_npz
import numpy as np
from tqdm import tqdm
import gc

from datetime import datetime

from joblib import Parallel, delayed

from skopt import Optimizer
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args

import wandb

space = [
    Integer(256, 1024, prior='log-uniform', name='k'),
    Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
    Integer(100, 1000, name='n_epochs'),
    Categorical(['Adam', 'AdamW', 'SGD'], name='optimizer')  # Optimizer choice
]

def sigmoid(x, clamp=False):
    if clamp:
        x = torch.clamp(x, min=-5, max=5)
    return 1./(1+ torch.exp(-x))

def log_(x):
    return torch.log(torch.clamp(x, min=1e-6))

def dot(x, y):
    return torch.mm(x, y.t())

def cosine(x, y):
    return torch.mm(x, y.t()) / (torch.norm(x, dim=-1) * torch.norm(y, dim=-1))

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

def e_step_val(q_logits,
               val_idx,
               U_left,
               U_right,
               bias,
               val_adj,
               dtype=torch.float32,
               device="cpu",
               lr=1e-2,
               n_iters=10):
    """
    Optimize only q_logits[val_idx] to maximize ELBO on val_adj,
    holding U_left, U_right, bias fixed.
    """
    # 1) extract and make it a Parameter
    q_val = nn.Parameter(q_logits[val_idx].clone().to(device))
    opt   = torch.optim.Adam([q_val], lr=lr)

    for _ in range(n_iters):
        q_prob = torch.softmax(q_val, dim=-1)   # (n_val × K)

        # build sparse A for val_adj (same as before)
        coo = val_adj.tocoo()
        idx = torch.stack([
            torch.from_numpy(coo.row).long(),
            torch.from_numpy(coo.col).long()
        ], dim=0).to(device)
        vals = torch.from_numpy(coo.data).to(dtype).to(device)
        A    = torch.sparse_coo_tensor(idx, vals, (len(val_idx), len(val_idx)), device=device)

        # compute edge/non-edge terms
        Aq      = torch.sparse.mm(A, q_prob)           # (n_val × K)
        edge_KK = q_prob.t() @ Aq                      # (K × K)
        s       = q_prob.sum(0)                        # (K,)
        all_KK  = s.unsqueeze(1) * s.unsqueeze(0)
        non_KK  = all_KK - edge_KK

        e_prob  = sigmoid(dot(U_left, U_right) + bias) # (K × K)
        obj     = (edge_KK * log_(e_prob)).sum() \
                + (non_KK   * log_(1-e_prob)).sum() \
                - (q_prob * log_(q_prob)).sum(1).mean()

        # gradient ascent on ELBO
        loss = -obj
        opt.zero_grad(); loss.backward(); opt.step()

    # 3) write back the inferred q_val
    with torch.no_grad():
        q_logits[val_idx] = q_val.data.cpu()

def compute_val_lowerbound(val_idx,
                          q_logits,
                          U_left,
                          U_right,
                          bias,
                          val_adj,
                          dtype=torch.float32,
                          device="cpu"):
    """
    Sparse‐only ELBO on held‐out nodes.
    """
    device = q_logits.device
    with torch.no_grad():
        # 1) posteriors on val nodes
        q = torch.softmax(q_logits[val_idx], dim=-1)  # (n_val × K)
        n_val, _ = q.shape

        # 2) convert SciPy CSR to PyTorch sparse_coo_tensor (one‐time)
        coo = val_adj.tocoo()
        idx = torch.stack([
            torch.from_numpy(coo.row).long(),
            torch.from_numpy(coo.col).long()
        ], dim=0).to(device)                            # (2 × E)
        vals = torch.from_numpy(coo.data).to(dtype).to(device)  # (E,)

        CC = torch.sparse_coo_tensor(idx, vals, (n_val, n_val), device=device)

        # 3) sparse‐dense multiply (n_val × K)
        # 4) accumulate K×K edge‐weight matrix
        edge_KK = q.t() @ torch.sparse.mm(CC, q)             # (K × K)

        # 5) non‐edge mass via total mass minus edges
        s = q.sum(dim=0)                   # (K,)
        all_pairs = s.unsqueeze(1) * s.unsqueeze(0) # (K × K)
        non_edge_KK = all_pairs - edge_KK

        # 6) compute ELBO exactly as before
        e_prob = sigmoid(dot(U_left, U_right) + bias)  # (K × K)
        edge_term = (edge_KK * log_(e_prob)).sum()
        non_term = (non_edge_KK * log_(1-e_prob)).sum()
        entropy = (q * log_(q)).sum(1).mean()
        obj = edge_term + non_term - entropy

        #avg_obj = obj / (n_val * n_val)
        #loss = -avg_obj
        loss = -obj

    # cleanup
    del CC, edge_KK, non_edge_KK, all_pairs, s, e_prob, edge_term, non_term, entropy, obj, q
    gc.collect()

    return loss.item()

# M-step: Update U_left and U_right
def m_step(N=None,
           n_max_updates=None,
           optimizer=None,
           minibatch_size=2500,
           lr=0.01,
           q_logits=None,
           U_left=None,
           U_right=None,
           bias=None,
           adj_matrix=None,
           dtype=torch.float32,
           device="cpu"):
    if q_logits is None or U_left is None or U_right is None or bias is None or adj_matrix is None:
        return
    if N is None:
        N = 1
    if n_max_updates is None:
        n_max_updates = 1

    if optimizer is None:
        optimizer = torch.optim.AdamW([U_left, U_right, bias, q_logits], lr=lr)

    initial_loss = 0.
    final_loss = 0.

    # create a random permutation of the indices
    indices_i = torch.randperm(N, device=device)
    indices_j = torch.randperm(N, device=device)
    n_minibatches = N // minibatch_size

    for i in tqdm(range(n_minibatches)):

        if i > n_max_updates:
            print(f"Reached maximum number of updates {n_max_updates}.")
            break

        # Get the minibatch indices
        minibatch_indices_i = indices_i[i * minibatch_size: (i + 1) * minibatch_size]
        minibatch_indices_j = indices_j[i * minibatch_size: (i + 1) * minibatch_size]

        # Get the minibatch of q_logits
        q_probs_batch_i = torch.softmax(q_logits[minibatch_indices_i], dim=-1) # mb_size x K 
        q_probs_batch_j = torch.softmax(q_logits[minibatch_indices_j], dim=-1) # mb_size x K

        # get the sub-matrix of adjacency matrix
        CC = torch.tensor(adj_matrix[minibatch_indices_i.cpu().numpy()]
                          [:, minibatch_indices_j.cpu().numpy()].toarray(), 
                          dtype=dtype).to(device) # mb_size x mb_size

        # outer product of q_logits_batch_i and q_logits_batch_j
        edge_weighted_KK = torch.mm(q_probs_batch_i.t(), torch.mm(CC, q_probs_batch_j)) # K x K
        non_edge_weighted_KK = torch.mm(q_probs_batch_i.t(), torch.mm(1-CC, q_probs_batch_j)) # K x K
        
        # compute the edge probabilities
        e_prob = sigmoid(dot(U_left, U_right) + bias) # K x K
        # e_prob = torch.clamp(e_prob, eps, 1 - eps)

        # compute the objective function
        obj = (edge_weighted_KK * log_(e_prob)).sum() + (non_edge_weighted_KK * log_(1 - e_prob)).sum()

        # add the entropy penalty on q_probs
        obj = obj - (q_probs_batch_i * log_(q_probs_batch_i)).sum(1).mean() / 2 
        obj = obj - (q_probs_batch_j * log_(q_probs_batch_j)).sum(1).mean() / 2

        # compute the loss
        loss = -obj

        # Zero the gradients
        optimizer.zero_grad()
        loss.backward()

        if i == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_([U_left, U_right, bias, q_logits], 1.0)

        # Perform a gradient step
        optimizer.step()
        del CC, e_prob, edge_weighted_KK, non_edge_weighted_KK, q_probs_batch_i, q_probs_batch_j, minibatch_indices_i, minibatch_indices_j
        gc.collect()

    # compute the loss
    #initial_loss = initial_loss / (minibatch_size * minibatch_size)
    #final_loss  = final_loss / (minibatch_size * minibatch_size)
    print(f"Initial loss: {initial_loss}, Final loss: {final_loss}")
    del U_left, U_right, bias, q_logits, optimizer, N, adj_matrix, dtype, minibatch_size, lr, indices_i, indices_j, n_minibatches, n_max_updates
    gc.collect()
    return initial_loss, final_loss

@use_named_args(space)
def objective(**params):
    run = wandb.init(
        project="flywrite1",
        resume="allow",
        reinit=True,           # allow multiple inits in the same process
        settings=wandb.Settings(start_method="thread")
    )

    K = params["k"]
    lr = params["learning_rate"]
    n_epochs = params["n_epochs"]
    optimizer_choice = params["optimizer"]
    
    device = "cpu"

    # Set the global seed using torch
    torch.manual_seed(int(datetime.now().timestamp()))

    # Constants
    d = 32       # Dimensionality of feature space
    minibatch_size = 2_500
    num_m_updates = 10_000  # Reduced for hyperparameter search

    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16

    # Load the sparse matrix
    adj_matrix = load_npz("./sparse_connectivity_matrix.npz")
    adj_matrix.data = (adj_matrix.data > 0).astype(np.int8)
    N = adj_matrix.shape[0]  # Number of nodes
    # split nodes once, get sub‐matrices
    train_idx, val_idx, train_adj, val_adj = train_val_split(adj_matrix, val_frac=0.2)

    N_train = train_idx.shape[0]
    print(f"Loaded adjacency matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.")

    # Model parameters
    U_left = nn.Parameter(1/np.sqrt(K * d) * torch.randn(K, d, dtype=dtype).to(device))
    U_right = nn.Parameter(1/np.sqrt(K * d) * torch.randn(K, d, dtype=dtype).to(device))
    bias = nn.Parameter(torch.log(torch.tensor([adj_matrix.mean()], dtype=dtype).to(device)))
    q_logits = nn.Parameter(1/K * torch.randn(N, K, dtype=dtype).to(device))

    # Create optimizer
    if optimizer_choice == 'Adam':
        optimizer = torch.optim.Adam([U_left, U_right, bias, q_logits], lr=lr)
    elif optimizer_choice == 'SGD':
        optimizer = torch.optim.SGD([U_left, U_right, bias, q_logits], lr=lr)
    else:
        optimizer = torch.optim.AdamW([U_left, U_right, bias, q_logits], lr=lr)

    val_elbo = 0

    del adj_matrix
    gc.collect()

    # --- TRAINING LOOP ---
    try:
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            # only train on the training subgraph
            initial_loss, final_loss = m_step(
                N=N_train,
                optimizer=optimizer,
                minibatch_size=minibatch_size,
                lr=lr,
                q_logits=q_logits,
                U_left=U_left,
                U_right=U_right,
                bias=bias,
                adj_matrix=train_adj,
                dtype=dtype,
                n_max_updates=num_m_updates
            )

            # --- VALIDATION STEP (OUTSIDE OF TRAINING LOOP) ---
            print("\n--- Validation Step ---")
            e_step_val(q_logits, val_idx, U_left, U_right, bias, val_adj, device=device)

            val_elbo = compute_val_lowerbound(
                val_idx=val_idx,
                q_logits=q_logits,
                U_left=U_left,
                U_right=U_right,
                bias=bias,
                val_adj=val_adj,
                dtype=dtype,
                device=device
            )
            print(f"Current Validation ELBO: {val_elbo}")
            run.log({
                "initial_train_elbo": initial_loss,
                "final_train_elbo": final_loss,
                "val_elbo": val_elbo,
            }, commit=True)

    except KeyboardInterrupt:
        print("Training interrupted.")
    
    print(f"Final Validation ELBO: {val_elbo}")
    
    # Clean up to avoid memory issues
    del optimizer, dtype, num_m_updates, minibatch_size, d
    gc.collect()
    run.finish()

    return -val_elbo  # Return negative ELBO for minimization

if __name__ == "__main__":
    # Set device
    device = "cpu"

    n_batches = 10
    batch_size = 5

    opt = Optimizer(dimensions=space, base_estimator="GP", acq_func="EI", random_state=42)

    for i in range(n_batches):
        candidates = opt.ask(n_points=batch_size)

        if candidates is None:
            raise ValueError("opt.ask() returned None")

        scores = Parallel(n_jobs=batch_size)(
            delayed(objective)(params) for params in candidates
        )
        
        opt.tell(candidates, scores)
        print(f"All configurations: {opt.Xi}")
        print(f"All scores: {opt.yi}")
        print(f"Batch {i+1}: Best score so far = {-min(opt.yi)}")

    # Best config
    best_idx = np.argmin(opt.yi)
    print("\nBest configuration:")
    print(f"  Params: {opt.Xi[best_idx]}")
    print(f"  Validation ELBO: {-opt.yi[best_idx]}")
