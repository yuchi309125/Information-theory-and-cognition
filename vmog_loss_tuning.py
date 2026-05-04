import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.sparse import load_npz
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score

from skopt import Optimizer
from skopt.space import Integer
from skopt.utils import use_named_args

import gc

# -----------------------------
# Search space (PCA removed)
# -----------------------------
# Only tune the number of GMM components (k).
space = [
    Integer(256, 1200, name='k'),
]


class BatchedEMGaussianMixture(nn.Module):
    def __init__(self, n_components, n_features, max_iter=100, tol=1e-3, reg_covar=1e-6, device='cpu'):
        """
        PyTorch implementation of Gaussian Mixture Model with batched EM algorithm.
        
        Args:
            n_components: Number of mixture components
            n_features: Number of features/dimensions in the data
            max_iter: Maximum number of EM iterations
            tol: Tolerance for convergence
            reg_covar: Regularization added to covariance matrices
            device: Device to use ('cpu', 'cuda', or 'mps')
        """
        super(BatchedEMGaussianMixture, self).__init__()
        
        self.n_components = n_components
        self.n_features = n_features
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.device = device
        
        # Learnable parameters (initialized in initialize_parameters)
        # These are buffers, not parameters, as we update them via EM not gradient descent
        self.register_buffer('means_', torch.zeros(n_components, n_features, device=device))
        self.register_buffer('covs_', torch.zeros(n_components, n_features, n_features, device=device))
        self.register_buffer('weights_', torch.ones(n_components, device=device) / n_components)
        
        # For diagonal covariance (more efficient)
        self.register_buffer('log_vars_', torch.zeros(n_components, n_features, device=device))
        
        # For tracking
        self.lower_bound_history_ = []
        self.n_iter_ = 0
        self.converged_ = False
        
        # For results
        self.responsibilities_ = None
        self.labels_ = None
        
    def initialize_parameters(self, X):
        """Initialize GMM parameters, optionally using random subset for stability."""
        n_samples = X.shape[0]
        
        # Random initialization
        if n_samples > 10000:
            # Sample a subset for more efficient initialization
            indices = torch.randperm(n_samples)[:10000]
            X_subset = X[indices]
        else:
            X_subset = X
            
        # Initialize means with random data points
        indices = torch.randperm(len(X_subset))[:self.n_components]
        self.means_ = X_subset[indices].clone()
        
        # Initialize with uniform weights
        self.weights_ = torch.ones(self.n_components, device=self.device) / self.n_components
        
        # Estimate initial variances from data
        data_var = torch.var(X_subset, dim=0)
        self.log_vars_ = torch.log(data_var + self.reg_covar).repeat(self.n_components, 1)

        # Clean up temporary tensors
        del X_subset, indices, data_var

        # Force garbage collection
        gc.collect()
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'mps') and self.device == 'mps':
            torch.mps.empty_cache()

    def e_step(self, X):
        """E-step: Compute responsibilities (posterior probabilities)."""
        n_samples = X.shape[0]
        log_resp = torch.zeros(n_samples, self.n_components, device=X.device)
        
        # Compute log probabilities for each component
        for k in range(self.n_components):
            # Using diagonal covariance for efficiency
            vars_k = torch.exp(self.log_vars_[k])
            
            # Compute log probabilities efficiently
            diff = X - self.means_[k]
            log_prob = -0.5 * (
                torch.sum(torch.log(2 * np.pi * vars_k)) + 
                torch.sum(diff**2 / vars_k.unsqueeze(0), dim=1)
            )
            
            log_resp[:, k] = torch.log(self.weights_[k] + 1e-10) + log_prob
            del diff, log_prob
        
        # Normalize (log-sum-exp trick for numerical stability)
        log_resp_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)
        log_resp = log_resp - log_resp_norm
        
        # Convert to probabilities
        resp = torch.exp(log_resp)

        # Clean up device tensors
        del log_resp, log_resp_norm
        
        # Force memory cleanup
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'mps') and self.device == 'mps':
            torch.mps.empty_cache()
        
        return resp
    
    def compute_lower_bound(self, X, resp):
        """Compute the lower bound (ELBO) for current parameters."""
        n_samples = X.shape[0]
        lower_bound = 0.0
        
        # Log-likelihood contribution
        for k in range(self.n_components):
            vars_k = torch.exp(self.log_vars_[k])
            
            diff = X - self.means_[k]
            log_prob = -0.5 * (
                torch.sum(torch.log(2 * np.pi * vars_k)) + 
                torch.sum(diff**2 / vars_k.unsqueeze(0), dim=1)
            )
            
            lower_bound += torch.sum(
                resp[:, k] * (torch.log(self.weights_[k] + 1e-10) + log_prob)
            )
            del vars_k, diff, log_prob
        
        # Entropy contribution
        entropy = -torch.sum(resp * torch.log(resp + 1e-10))
        lower_bound += entropy

        # Clean up device tensors
        del entropy
        # Force cleanup
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'mps') and self.device == 'mps':
            torch.mps.empty_cache()
        
        return lower_bound / n_samples
    
    def fit(self, X, batch_size=1024, verbose=False):
        """
        Fit the GMM using batched EM algorithm.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            batch_size: Size of batches for processing
            verbose: Whether to print progress
            
        Returns:
            self: Fitted model
        """
        n_samples = X.shape[0]
        
        # Move data to the right device if needed
        if X.device != self.device:
            X = X.to(self.device)
        
        # Initialize parameters
        self.initialize_parameters(X)
        
        # Store for convergence check
        prev_lower_bound = -np.inf
        prev_means = self.means_.clone()
        
        # Create data loader for batched processing
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for iteration in range(self.max_iter):
            # Initialize accumulators for sufficient statistics
            nk = torch.zeros(self.n_components, device=self.device)
            means_numerator = torch.zeros_like(self.means_)
            vars_numerator = torch.zeros_like(self.log_vars_)
            
            # For ELBO calculation
            total_lower_bound = 0.0
            
            # Process batches
            n_processed = 0
            for batch_idx, (batch_X,) in enumerate(tqdm(loader, desc=f"EM Iteration {iteration+1}")):
                batch_size_actual = batch_X.shape[0]
                n_processed += batch_size_actual
                
                # E-step: compute responsibilities for this batch
                with torch.no_grad():
                    batch_resp = self.e_step(batch_X)
                
                # Accumulate statistics for M-step
                batch_nk = torch.sum(batch_resp, dim=0)
                nk += batch_nk
                
                # Means numerator: Σ_i r_ik * x_i
                means_numerator += torch.matmul(batch_resp.T, batch_X)
                
                # Variances numerator: Σ_i r_ik * (x_i - μ_k)^2
                for k in range(self.n_components):
                    diff = batch_X - self.means_[k]
                    # Weighted sum of squared differences
                    weighted_diff_sq = batch_resp[:, k].unsqueeze(1) * diff**2
                    vars_numerator[k] += torch.sum(weighted_diff_sq, dim=0)
                    del diff, weighted_diff_sq
                
                # Contribution to lower bound
                batch_lower_bound = self.compute_lower_bound(batch_X, batch_resp)
                total_lower_bound += batch_lower_bound * batch_size_actual
                
                # Clean up between batches
                del batch_resp, batch_nk
                if hasattr(torch, 'mps') and torch.backends.mps.is_available() and self.device == 'mps':
                    torch.mps.empty_cache()
                elif torch.cuda.is_available() and self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            # M-step: update parameters using accumulated statistics
            with torch.no_grad():
                # Update means: μ_k = (Σ_i r_ik * x_i) / (Σ_i r_ik)
                self.means_ = means_numerator / nk.unsqueeze(1)
                
                # Update variances: σ²_k = (Σ_i r_ik * (x_i - μ_k)²) / (Σ_i r_ik)
                self.log_vars_ = torch.log(vars_numerator / nk.unsqueeze(1) + self.reg_covar)
                
                # Update weights: π_k = (Σ_i r_ik) / N
                self.weights_ = nk / n_samples
            
            # Normalize lower bound
            total_lower_bound /= n_samples
            self.lower_bound_history_.append(total_lower_bound.item())
            
            if verbose:
                print(f"Iteration {iteration+1}: Lower bound = {total_lower_bound.item():.4f}")
            
            # Check for convergence
            if iteration > 0:
                mean_change = torch.mean(torch.abs(self.means_ - prev_means))
                lb_change = total_lower_bound - prev_lower_bound
                
                if verbose:
                    print(f"Mean change: {mean_change.item():.6f}, LB change: {lb_change.item():.6f}")
                
                if (mean_change < self.tol or lb_change < self.tol) and lb_change >= 0:
                    self.converged_ = True
                    self.n_iter_ = iteration + 1
                    if verbose:
                        print(f"Converged after {self.n_iter_} iterations")
                    break
            
            prev_lower_bound = total_lower_bound
            prev_means = self.means_.clone()
        
        # If not converged, set final iteration count
        if not self.converged_:
            self.n_iter_ = self.max_iter
            if verbose:
                print(f"Did not converge after {self.max_iter} iterations")
        
        # Compute final responsibilities and labels in batches
        self.responsibilities_ = torch.zeros(n_samples, self.n_components, device='cpu')
        self.labels_ = torch.zeros(n_samples, dtype=torch.long, device='cpu')
        
        with torch.no_grad():
            start_idx = 0
            for batch_idx, (batch_X,) in enumerate(tqdm(loader, desc="Computing final assignments")):
                batch_size_actual = batch_X.shape[0]
                batch_resp = self.e_step(batch_X)
                
                # Move to CPU for storage (to save GPU memory)
                self.responsibilities_[start_idx:start_idx+batch_size_actual] = batch_resp.cpu()
                self.labels_[start_idx:start_idx+batch_size_actual] = torch.argmax(batch_resp, dim=1).cpu()
                
                start_idx += batch_size_actual
        
        # Analysis of clustering results (optional printouts)
        unique_labels, counts = torch.unique(self.labels_, return_counts=True)
        if verbose:
            print(f"Found {len(unique_labels)} unique clusters out of {self.n_components} components")
        
        return self


@use_named_args(space)
def objective(**params):
    """
    Objective for Bayesian optimization (silhouette score).
    PCA steps have been removed. We run GMM directly on the raw (dense) features.
    """
    n_mog_components = params["k"]

    # Load the sparse matrix
    file_path = "./sparse_connectivity_matrix.npz"
    adj_matrix = load_npz(file_path)
    print(f"Loaded sparse matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.")

    # Convert to dense (NOTE: may be memory intensive depending on data size)
    X_np = adj_matrix.toarray().astype(np.float32)
    del adj_matrix
    gc.collect()

    # Create torch tensor
    X = torch.tensor(X_np, dtype=torch.float32, device=device)

    # Create and fit GMM directly on raw features
    gmm_model = BatchedEMGaussianMixture(
        n_components=n_mog_components,
        n_features=X.shape[1],
        max_iter=100,
        tol=1e-4,
        reg_covar=1e-6,
        device=device
    )

    gmm_model.fit(X, batch_size=1024, verbose=True)

    # Get labels - already computed during fitting
    labels = gmm_model.labels_

    # Compute the silhouette score (subsample if too large for speed/memory)
    labels_np = labels.numpy()
    max_samples_for_silhouette = 20000
    if X_np.shape[0] > max_samples_for_silhouette:
        idx = np.random.choice(X_np.shape[0], max_samples_for_silhouette, replace=False)
        score = silhouette_score(X_np[idx], labels_np[idx])
    else:
        score = silhouette_score(X_np, labels_np)

    # Cleanup
    del labels, X, X_np
    gc.collect()
    if hasattr(torch, 'mps') and device == 'mps':
        torch.mps.empty_cache()
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Optimizer minimizes the objective, so return negative silhouette score.
    return -score


if __name__ == "__main__":
    # Set device (global for objective)
    device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Basic optimizer loop (kept structure, defined missing vars)
    n_batches = 3          # you can adjust
    batch_size = 4         # number of candidates per batch

    opt = Optimizer(dimensions=space, base_estimator="GP", acq_func="EI", random_state=42)

    for i in range(n_batches):
        candidates = opt.ask(n_points=batch_size)
        if candidates is None:
            raise ValueError("opt.ask() returned None")

        scores = Parallel(n_jobs=batch_size)(
            delayed(objective)(params) for params in candidates
        )
       
        opt.tell(candidates, scores)
        print(f"All scores so far = {opt.yi}")
        print(f"Batch {i+1}: Best score so far = {-min(opt.yi):.4f}")

    # Best config
    best_idx = np.argmin(opt.yi)
    print("\nBest configuration:")
    print(f"  Params: {opt.Xi[best_idx]}")
    print(f"  Silhouette Score: {-opt.yi[best_idx]:.4f}")
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmog_loss_tuning.py
---------------------------------
Loss-based (SBM-style) tuning to replace PCA in the clustering pipeline.

- Learns soft cluster assignments q (N × K), a block affinity matrix C (K × K),
  and a global bias b by maximizing an edge likelihood on a train split.
- Uses a held-out validation split to compute an ELBO-style objective (avg log-likelihood).
- Hyperparameters are tuned with Bayesian optimization (skopt).
- Produces hard labels via argmax over q after training with the best configuration.

This file is self-contained and does NOT use PCA.
"""

import os
import gc
import json
import math
import shutil
import random
import pickle
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scipy.sparse import load_npz, coo_matrix
from joblib import Parallel, delayed
from tqdm import tqdm

from skopt import Optimizer
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args


# ---------------------------- Utils ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_empty_cache(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def estimate_max_jobs_per_gpu(required_per_job_mb=4000):
    """
    Estimate the maximum number of jobs that can run concurrently on the available GPU memory
    assuming each job requires ~4000MB by default
    """
    if not torch.cuda.is_available():
        return 1  # CPU fallback

    total_free = 0
    for i in range(torch.cuda.device_count()):
        stats = torch.cuda.mem_get_info(i)
        free_mb = stats[0] / (1024 ** 2)  # bytes → MB
        total_free += free_mb

    max_jobs = max(int(total_free // required_per_job_mb), 1)
    return max_jobs


# ---------------------------- Data split ----------------------------

def train_val_split(adj: coo_matrix, val_frac: float = 0.2, seed: int = 42):
    """Return index split and corresponding subgraphs (COO)."""
    set_seed(seed)
    N = adj.shape[0]
    perm = np.random.permutation(N)
    n_val = int(N * val_frac)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    train_adj = adj.tocsr()[train_idx][:, train_idx].tocoo()
    val_adj = adj.tocsr()[val_idx][:, val_idx].tocoo()
    return train_idx, val_idx, train_adj, val_adj


# ---------------------------- SBM-style model ----------------------------

@dataclass
class SbmConfig:
    K: int
    lr: float = 1e-2
    n_epochs: int = 500
    batch_edges: int = 4096
    neg_ratio: float = 1.0         # negatives per positive edge
    optimizer: str = "Adam"        # 'Adam' | 'AdamW' | 'SGD'
    val_steps: int = 30            # steps to optimize q on validation (E-step on val)
    device: str = "cpu"
    seed: int = 42


class SbmModel(nn.Module):
    """
    Simple symmetric SBM with logistic link:
        p(A_ij=1) = sigmoid( q_i^T C q_j + b )
    - q_logits: (N, K) free logits, q = softmax(q_logits)
    - C: (K, K) block affinity (symmetric)
    - b: scalar bias
    """
    def __init__(self, N: int, config: SbmConfig):
        super().__init__()
        self.N = N
        self.config = config

        self.q_logits = nn.Parameter(torch.zeros(N, config.K))
        # symmetric C initialization
        C = 0.01 * torch.randn(config.K, config.K)
        C = 0.5 * (C + C.t())
        self.C = nn.Parameter(C)
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _bce_log_prob(logit, y):
        # log p = y*logσ + (1-y)*log(1-σ) = -BCE
        return -nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")

    def forward_logits(self, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
        # logits for edges (i,j)
        q = nn.functional.softmax(self.q_logits, dim=-1)  # (N, K)
        qi = q[i_idx]   # (B, K)
        qj = q[j_idx]   # (B, K)
        # bilinear form qi^T C qj
        z = (qi @ self.C) * qj
        z = z.sum(dim=-1) + self.bias  # (B,)
        return z

    def elbo_batch(self, i_idx, j_idx, y):
        logits = self.forward_logits(i_idx, j_idx)
        return self._bce_log_prob(logits, y).mean()

    @torch.no_grad()
    def hard_labels(self) -> np.ndarray:
        q = nn.functional.softmax(self.q_logits, dim=-1)  # (N,K)
        return torch.argmax(q, dim=-1).cpu().numpy()


# ---------------------------- Sampling ----------------------------

def sample_edges(adj: coo_matrix, num_samples: int, rng: np.random.Generator):
    """Sample positive edges (i,j) uniformly from non-zeros (upper triangle if symmetric)."""
    # Ensure COO
    adj = adj.tocoo()
    rows, cols = adj.row, adj.col

    # If graph is symmetric, sample only i<j
    mask = rows < cols
    pos_r = rows[mask]
    pos_c = cols[mask]
    M = len(pos_r)
    if M == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    idx = rng.integers(0, M, size=min(num_samples, M), endpoint=False)
    return pos_r[idx], pos_c[idx]


def sample_negatives(N: int, num_samples: int, rng: np.random.Generator):
    """Uniformly sample node pairs as negatives. In sparse graphs, collision with real edges is rare."""
    i = rng.integers(0, N, size=num_samples, endpoint=False)
    j = rng.integers(0, N, size=num_samples, endpoint=False)
    # Avoid i==j
    same = (i == j)
    if same.any():
        j[same] = (j[same] + 1) % N
    return i, j


# ---------------------------- Training & Validation ----------------------------

def train_sbm(adj: coo_matrix, config: SbmConfig) -> Tuple[SbmModel, float, float]:
    """
    Train SBM on 'adj' with negative sampling.
    Returns: (model, initial_train_elbo, final_train_elbo)
    """
    device = config.device
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    N = adj.shape[0]
    model = SbmModel(N, config).to(device)

    if config.optimizer == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == "AdamW":
        opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=config.lr)

    # Precompute positives
    pos_r_all, pos_c_all = sample_edges(adj, adj.nnz, rng)  # may be large; we'll subsample inside loop
    initial_elbo, final_elbo = None, None

    for epoch in tqdm(range(config.n_epochs), desc="SBM-Train"):
        # Mini-batch: subsample positives, then negatives
        pos_r, pos_c = sample_edges(adj, config.batch_edges, rng)
        if pos_r.size == 0:
            break
        neg_num = int(config.batch_edges * config.neg_ratio)
        neg_r, neg_c = sample_negatives(N, neg_num, rng)

        # Build tensors
        i_idx = torch.from_numpy(np.concatenate([pos_r, neg_r])).long().to(device)
        j_idx = torch.from_numpy(np.concatenate([pos_c, neg_c])).long().to(device)
        y = torch.cat([torch.ones(len(pos_r)), torch.zeros(len(neg_r))]).float().to(device)

        opt.zero_grad(set_to_none=True)
        loss = -model.elbo_batch(i_idx, j_idx, y)  # maximize ELBO -> minimize -ELBO
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if epoch == 0:
            initial_elbo = -loss.item()

        if (epoch + 1) % 50 == 0:
            final_elbo = -loss.item()

    if final_elbo is None:
        final_elbo = initial_elbo if initial_elbo is not None else 0.0

    gc.collect()
    safe_empty_cache(device)
    return model, float(initial_elbo), float(final_elbo)


@torch.no_grad()
def val_elbo(adj_val: coo_matrix, model: SbmModel, steps: int = 30) -> float:
    """
    Compute validation ELBO with a local E-step on q for validation nodes,
    keeping C and bias fixed.
    """
    device = next(model.parameters()).device
    N_val = adj_val.shape[0]
    K = model.config.K

    # Clone a local copy of q_logits for val nodes
    q_logits_val = nn.Parameter(torch.zeros(N_val, K, device=device))
    opt = torch.optim.SGD([q_logits_val], lr=0.1)

    # Prepare positive and negative pairs
    rng = np.random.default_rng(12345)
    pos_r, pos_c = sample_edges(adj_val, min(adj_val.nnz, 50000), rng)
    if pos_r.size == 0:
        return -1e9  # no edges -> very bad
    neg_r, neg_c = sample_negatives(N_val, len(pos_r), rng)

    i_idx = torch.from_numpy(np.concatenate([pos_r, neg_r])).long().to(device)
    j_idx = torch.from_numpy(np.concatenate([pos_c, neg_c])).long().to(device)
    y = torch.cat([torch.ones(len(pos_r)), torch.zeros(len(neg_r))]).float().to(device)

    # Temporary module that reuses model.C and bias but with local q
    class _ValWrap(nn.Module):
        def __init__(self, C, bias, q_logits_ref):
            super().__init__()
            self.C = C
            self.bias = bias
            self.q_logits_ref = q_logits_ref
        def forward_logits(self, i_idx, j_idx):
            q = nn.functional.softmax(self.q_logits_ref, dim=-1)
            qi = q[i_idx]; qj = q[j_idx]
            z = (qi @ self.C) * qj
            return z.sum(dim=-1) + self.bias

    wrap = _ValWrap(model.C, model.bias, q_logits_val)

    # Optimize q on validation nodes
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = wrap.forward_logits(i_idx, j_idx)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()

    # Compute average log-likelihood (ELBO approximation)
    logits = wrap.forward_logits(i_idx, j_idx)
    ll = -nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="mean")
    return float(ll.item())


# ---------------------------- Hyperparameter search ----------------------------

# Search space (you can widen as needed)
space = [
    Integer(64, 2048, name='k'),               # number of clusters
    Real(1e-4, 5e-2, prior='log-uniform', name='learning_rate'),
    Integer(200, 1500, name='n_epochs'),
    Integer(1024, 16384, name='batch_edges'),
    Real(0.5, 2.0, name='neg_ratio'),
    Categorical(['Adam', 'AdamW', 'SGD'], name='optimizer'),
]


@use_named_args(space)
def objective(**params):
    """Minimize negative validation ELBO (maximize ELBO)."""
    # Load graph
    file_path = "./sparse_connectivity_matrix.npz"
    adj = load_npz(file_path).tocsr()
    N = adj.shape[0]
    # ensure symmetry & make COO
    adj = (adj + adj.T).multiply(0.5).tocsr()
    adj_coo = adj.tocoo()

    # Split
    train_idx, val_idx, train_adj, val_adj = train_val_split(adj_coo, val_frac=0.2, seed=42)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(0)

    # Config
    cfg = SbmConfig(
        K=int(params["k"]),
        lr=float(params["learning_rate"]),
        n_epochs=int(params["n_epochs"]),
        batch_edges=int(params["batch_edges"]),
        neg_ratio=float(params["neg_ratio"]),
        optimizer=str(params["optimizer"]),
        val_steps=30,
        device=device,
        seed=42,
    )

    # Train on train graph
    model, init_elbo, final_elbo = train_sbm(train_adj, cfg)

    # Validate
    vll = val_elbo(val_adj, model, steps=cfg.val_steps)
    print(f"[VAL] ELBO={vll:.6f}  (init={init_elbo:.6f}, final={final_elbo:.6f})")

    # Cleanup
    del model, adj, adj_coo, train_adj, val_adj
    gc.collect(); safe_empty_cache(device)

    return -vll  # skopt minimizes


def main():
    set_seed(42)
    total_trials = 20  # adjust as needed

    # Parallel policy: 1 per GPU, else 1 job
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_jobs_per_batch = max(1, gpu_count)
    n_batches = (total_trials + n_jobs_per_batch - 1) // n_jobs_per_batch

    opt = Optimizer(dimensions=space, base_estimator="GP", acq_func="EI", random_state=42)
    space_names = [d.name for d in space]

    best_loss = float("inf")
    best_cfg = None

    for bi in range(n_batches):
        candidates = opt.ask(n_points=n_jobs_per_batch)
        if not isinstance(candidates, list):
            candidates = [candidates]

        def run_one(point):
            params = {name: val for name, val in zip(space_names, point)}
            return objective(**params)

        scores = Parallel(n_jobs=n_jobs_per_batch)(delayed(run_one)(pt) for pt in candidates)
        opt.tell(candidates, scores)

        cur_best = min(opt.yi)
        if cur_best < best_loss:
            best_loss = cur_best
            best_point = opt.Xi[int(np.argmin(opt.yi))]
            best_cfg = {name: val for name, val in zip(space_names, best_point)}

        print(f"[Batch {bi+1}/{n_batches}] best ELBO so far = {-best_loss:.6f}")

    print("\nBest configuration:")
    print(best_cfg)
    print(f"Best validation ELBO: {-best_loss:.6f}")

    # --- Train final model on FULL graph with best config and save labels ---
    file_path = "./sparse_connectivity_matrix.npz"
    adj_full = load_npz(file_path).tocsr()
    adj_full = (adj_full + adj_full.T).multiply(0.5).tocsr().tocoo()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(0)

    cfg = SbmConfig(
        K=int(best_cfg["k"]),
        lr=float(best_cfg["learning_rate"]),
        n_epochs=int(best_cfg["n_epochs"]),
        batch_edges=int(best_cfg["batch_edges"]),
        neg_ratio=float(best_cfg["neg_ratio"]),
        optimizer=str(best_cfg["optimizer"]),
        val_steps=30,
        device=device,
        seed=42,
    )

    model, init_elbo, final_elbo = train_sbm(adj_full, cfg)
    labels = model.hard_labels()

    os.makedirs("vmog_runs", exist_ok=True)
    np.save("vmog_runs/sbm_labels_FINAL.npy", labels)
    torch.save({
        "q_logits": model.q_logits.detach().cpu(),
        "C": model.C.detach().cpu(),
        "bias": model.bias.detach().cpu(),
        "config": cfg.__dict__,
        "best_cfg": best_cfg,
        "best_val_elbo": float(-best_loss),
    }, "vmog_runs/sbm_model_FINAL.pt")
    print("[INFO] Saved labels to vmog_runs/sbm_labels_FINAL.npy and model to vmog_runs/sbm_model_FINAL.pt")


if __name__ == "__main__":
    main()
