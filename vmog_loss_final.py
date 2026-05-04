
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path
import gc
import json

# ------------------------------------
# Utilities
# ------------------------------------

def load_mapping(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    # mapping should map row index -> root_id
    return mapping

def sigmoid(x, clamp=False):
    if clamp:
        x = torch.clamp(x, min=-5, max=5)
    return 1.0 / (1.0 + torch.exp(-x))

# ------------------------------------
# Mini-batch GMM with sbm-style M-step
# ------------------------------------

class MiniBatchGMM(nn.Module):
    """GMM trained via generalized EM in a streaming fashion:
      - E-step: compute responsibilities for the current mini-batch
      - M-step: gradient ascent on expected complete-data log-likelihood
                using only the current mini-batch (sbm.py style)
    Diagonal covariance; parameters are updated with an optimizer.
    """
    def __init__(self, n_components, n_features, device='cpu', dtype=torch.float32):
        super().__init__()
        self.K = n_components
        self.D = n_features
        self.device = device
        self.dtype = dtype

        # Parameters to learn (initialized later)
        self.means = nn.Parameter(torch.zeros(self.K, self.D, dtype=dtype, device=device))
        # log-variance per component & feature (diagonal covariance)
        self.log_vars = nn.Parameter(torch.zeros(self.K, self.D, dtype=dtype, device=device))
        # unnormalized logits for mixture weights
        self.logits = nn.Parameter(torch.zeros(self.K, dtype=dtype, device=device))

        # buffers for results
        self.labels_ = None

    def initialize_params_from_sample(self, X_init):
        device = X_init.device
        K = self.K
        D = X_init.shape[1]

        N0 = X_init.shape[0]
        if N0 >= K:
            idx = torch.randperm(N0, device=device)[:K]
        else:
            idx = torch.randint(0, N0, (K,), device=device)
        mu0 = X_init[idx]  # (K, D)

        eps = 1e-6
        var = X_init.var(dim=0, unbiased=False) + eps        # (D,)
        init_log_vars = var.log().unsqueeze(0).repeat(K, 1).contiguous()  # (K, D)

        with torch.no_grad():
            self.means.copy_(mu0)           # (K, D)
            self.log_vars.copy_(init_log_vars)
            self.logits.zero_()


    def e_step_batch(self, Xb):
        """Compute responsibilities for a batch.
        Xb: (B, D) on device -> returns: resp (B, K)"""
        K, D = self.K, self.D
        B = Xb.size(0)
        log_resp = torch.empty(B, K, dtype=self.dtype, device=self.device)
        log_pi = F.log_softmax(self.logits, dim=0)  # (K,)
        inv_vars = torch.exp(-self.log_vars)        # (K, D)
        const = -0.5 * D * np.log(2 * np.pi)
        for k in range(K):
            diff = Xb - self.means[k].unsqueeze(0)           # (B, D)
            term1 = const - 0.5 * torch.sum(self.log_vars[k])  # scalar
            term2 = -0.5 * torch.sum(diff * diff * inv_vars[k].unsqueeze(0), dim=1)  # (B,)
            log_prob = term1 + term2                           # (B,)
            log_resp[:, k] = log_pi[k] + log_prob
        log_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)
        resp = torch.exp(log_resp - log_norm)  # (B, K)
        return resp

    def expected_complete_loglik_batch(self, Xb, resp):
        """Compute expected complete-data log-likelihood for batch (sum over batch)."""
        K, D = self.K, self.D
        log_pi = F.log_softmax(self.logits, dim=0)  # (K,)
        inv_vars = torch.exp(-self.log_vars)        # (K, D)
        elbo = torch.zeros((), dtype=self.dtype, device=self.device)
        const = -0.5 * D * np.log(2 * np.pi)
        for k in range(K):
            diff = Xb - self.means[k].unsqueeze(0)  # (B, D)
            quad = -0.5 * torch.sum(diff * diff * inv_vars[k].unsqueeze(0), dim=1)  # (B,)
            log_det = -0.5 * torch.sum(self.log_vars[k])  # scalar
            log_comp = log_pi[k] + const + log_det + quad  # (B,)
            elbo += torch.sum(resp[:, k] * log_comp)  # scalar
        return elbo

    def fit_streaming(self,
                      csr_X: csr_matrix,
                      batch_size=64,
                      n_epochs=3,
                      m_steps_per_batch=1,
                      lr=1e-2,
                      verbose=True):
        """Train with streaming mini-batches from a CSR sparse matrix (rows are samples)."""
        N, D = csr_X.shape
        assert D == self.D

        # init
        init_idx = np.random.default_rng(42).choice(N, size=min(512, N), replace=False)
        X_init = torch.tensor(csr_X[init_idx].toarray(), dtype=self.dtype, device=self.device)
        self.initialize_params_from_sample(X_init)
        del X_init; gc.collect()

        opt = torch.optim.Adam([self.means, self.log_vars, self.logits], lr=lr)

        for epoch in range(n_epochs):
            perm = np.random.permutation(N)
            for t in tqdm(range(0, N, batch_size), desc=f"Epoch {epoch+1}/{n_epochs}"):
                idx = perm[t: t+batch_size]
                Xb_np = csr_X[idx].toarray().astype(np.float32)
                Xb = torch.tensor(Xb_np, dtype=self.dtype, device=self.device)

                # E-step for batch
                with torch.no_grad():
                    resp = self.e_step_batch(Xb)  # (B, K)

                # M-step (sbm-style): gradient ascent on expected complete-data loglik
                for _ in range(m_steps_per_batch):
                    opt.zero_grad(set_to_none=True)
                    elbo = self.expected_complete_loglik_batch(Xb, resp)
                    loss = -elbo
                    loss.backward()
                    opt.step()

                # free batch
                del Xb, Xb_np, resp, loss, elbo
                gc.collect()
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return self

    def predict_streaming(self, csr_X: csr_matrix, batch_size=1024):
        """Assign labels in a streaming way; returns (N,) long tensor on CPU."""
        N, D = csr_X.shape
        labels = []
        for t in tqdm(range(0, N, batch_size), desc="Assigning clusters"):
            idx = slice(t, min(t+batch_size, N))
            Xb_np = csr_X[idx].toarray().astype(np.float32)
            Xb = torch.tensor(Xb_np, dtype=self.dtype, device=self.device)
            with torch.no_grad():
                resp = self.e_step_batch(Xb)
                lab = torch.argmax(resp, dim=1).cpu()
            labels.append(lab)
            del Xb, Xb_np, resp, lab
            gc.collect()
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.labels_ = torch.cat(labels, dim=0)
        return self.labels_


# ------------------------------------
# PCA-free run function (streaming)
# ------------------------------------

def run_pca_gmm(batch, index, device="cuda",
                K=967,   # n_components
                epochs=3,
                batch_size=64,
                m_steps_per_batch=1,
                lr=1e-2):
    """PCA-free, streaming version using sbm.py-style M-step (gradient ascent on ELBO)."""
    torch_dtype = torch.float32 if device != 'cuda' else torch.bfloat16

    iteration = (batch * 4) + index

    # load CSR sparse features
    csr_X = load_npz("./sparse_connectivity_matrix.npz").tocsr()
    N, D = csr_X.shape
    print(f"[run_pca_gmm-noPCA/stream] CSR shape={N}x{D}, nnz={csr_X.nnz}")

    model = MiniBatchGMM(n_components=K, n_features=D, device=device, dtype=torch_dtype)

    model.fit_streaming(csr_X, batch_size=batch_size, n_epochs=epochs,
                        m_steps_per_batch=m_steps_per_batch, lr=lr, verbose=True)

    labels = model.predict_streaming(csr_X, batch_size=1024)   # (N,)

    Path("vmog_runs").mkdir(parents=True, exist_ok=True)
    torch.save(model.means.detach().cpu().float(), f"vmog_runs/vmog_cluster_centers_tuned_{iteration}.pt")
    torch.save(labels.cpu(), f"vmog_runs/vmog_labels_tuned_{iteration}.pt")

    # optional mapping save
    try:
        with open('./root_id_to_index_mapping.json','r',encoding='utf-8') as f:
            mapping = json.load(f)
        if isinstance(mapping, dict):
            cluster_assignment_dict = { mapping.get(str(i), i): int(labels[i].item()) for i in range(len(labels)) }
        else:
            cluster_assignment_dict = { mapping[i]: int(labels[i].item()) for i in range(len(labels)) }
        import numpy as _np
        _np.save(f"vmog_runs/vmog_cluster_assignment_dict_tuned_{iteration}.npy", cluster_assignment_dict, allow_pickle=True)
    except Exception as e:
        print("[warn] mapping not saved:", e)

    print("Done.")
    return labels


# if __name__ == '__main__':
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     batch_size = 4
#     n_batches = 12
#     for batch in range(n_batches):
#         _ = Parallel(n_jobs=batch_size)(
#             delayed(run_pca_gmm)(batch, index=i+1, device=device)
#             for i in range(batch_size)
#         )




if __name__ == '__main__':
    import argparse, os
    from joblib import Parallel, delayed
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0', help='0,1,2,3')
    parser.add_argument('--jobs', type=int, default=1, help='GPU number')
    parser.add_argument('--n_batches', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=4, help='number of batches per run')
    parser.add_argument('--K', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--minibatch', type=int, default=64)
    parser.add_argument('--msteps', type=int, default=1)
    args = parser.parse_args()

    gpu_ids = [g.strip() for g in args.gpus.split(',') if g.strip() != '']
    use_cuda = torch.cuda.is_available() and (len(gpu_ids) > 0) and (gpu_ids[0].lower() != 'cpu')
    n_devices = len(gpu_ids) if use_cuda else 1
    max_jobs = n_devices if use_cuda else 1
    n_jobs = min(args.jobs, max_jobs)

    def _runner(batch_idx, task_idx):
        if use_cuda:
            gid = int(gpu_ids[(task_idx - 1) % n_devices]) 
            device = f'cuda:{gid}'
            torch.cuda.set_device(gid)
        else:
            device = 'cpu'
        return run_pca_gmm(
            batch=batch_idx,
            index=task_idx,
            device=device,
            K=args.K,
            epochs=args.epochs,
            batch_size=args.minibatch,
            m_steps_per_batch=args.msteps,
        )

    for batch in range(args.n_batches):
        _ = Parallel(n_jobs=n_jobs)(
            delayed(_runner)(batch, i + 1) for i in range(args.batch_size)
        )
