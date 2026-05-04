import torch
from scipy.sparse import csr_matrix, load_npz
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score
import numpy as np
from tqdm import tqdm
import wandb as wdb
from joblib import Parallel, delayed
import gc
import multiprocessing as mp

from ignite.engine import Engine
from ignite.metrics.clustering import SilhouetteScore

from skopt import Optimizer, gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args

# build a index-to-root_id dictionary
from index_mapping import load_mapping


def stochastic_pca(W_csr, n_components, batch_size=128, lr=0.01, max_iter=1000, tol=1e-6, optimizer_choice="Adam",device="cuda"):
    """
    Perform stochastic PCA on a sparse matrix to minimize || U U^T W - W ||^2_F.

    Args:
        W_csr: Sparse matrix (csr_matrix).
        n_components: Number of principal components (d).
        batch_size: Number of rows to sample in each iteration.
        lr: Learning rate for Adam optimizer.
        max_iter: Maximum number of iterations.
        tol: Tolerance for convergence.
        device: Device to use ("cuda" or "cpu").

    Returns:
        U: Learned principal components matrix (N x d).
    """
    print("Initializing stochastic PCA...")
    print("optimizer", optimizer_choice)

    N, M = W_csr.shape  # Number of rows (N) and columns (features, M)
    d = n_components

    # Initialize U randomly
    U = torch.randn(N, d, device=device, requires_grad=True)  # Enable gradients for U
    # Initialize b randomly
    b = torch.tensor([W_csr.mean()], dtype=torch.float32).to(device)

    # Convert W_csr to coordinate format for efficient access
    # W_coo = W_csr.tocoo()

    # Initialize Adam optimizer
    optimizer = torch.optim.Adam([U, b], lr=lr)
    if optimizer_choice == 'Adam':
        optimizer = torch.optim.Adam([U, b], lr=lr)
    elif optimizer_choice == 'SGD':
        optimizer = torch.optim.SGD([U, b], lr=lr)
    elif optimizer_choice == 'AdamW':
        optimizer = torch.optim.AdamW([U, b], lr=lr)

    try:
        # Optimization loop
        for it in tqdm(range(max_iter)):
            # Randomly sample a batch of rows
            batch_indices = np.random.choice(N, size=batch_size, replace=False)

            # Extract the batch rows from W using SciPy's indexing
            W_batch_csr = W_csr[batch_indices]  # Still sparse
            W_batch = torch.tensor(W_batch_csr.toarray(), dtype=torch.float32, device=device)  # Dense batch
            U_batch = U[batch_indices]  # Corresponding rows of U

            # Compute reconstruction: U_batch @ (U.T @ W_batch)
            UT_W = torch.mm(W_batch, U)  # Shape: (batch_size, d)
            W_reconstructed = torch.mm(UT_W, U.T)  # Shape: (batch_size, features)

            # Compute reconstruction error for the batch
            diff = W_reconstructed - W_batch + b
            loss = torch.norm(diff, p='fro') ** 2

            # Zero the gradients
            optimizer.zero_grad()

            # Backpropagate the loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_([U, b], max_norm=1.0)

           # wdb.log({
           #     "lr": get_lr(optimizer),
           #     "iteration": it,
           #     "bias": b[0],
           #     "loss": loss,
           #     "U_batch": torch.norm(U[batch_indices]),
           #     "UT_W": torch.norm(UT_W),
           #     "W_batch": torch.norm(W_batch),
           #     "W_reconstructed": torch.norm(W_reconstructed),
           # })

            # Perform an Adam optimization step
            optimizer.step()

            del batch_indices, W_batch_csr, W_batch, U_batch, UT_W, W_reconstructed, diff

#             # Re-normalize U after the update
#            with torch.no_grad():
#                U_batch_norm = torch.norm(U_batch, dim=1, keepdim=True)
#                U[batch_indices] = U_batch / U_batch_norm.clamp(min=1e-8)  # Prevent division by zero

            if it % 50 == 0:
                gc.collect()
                if torch.backends.mps.is_available() and torch.device('mps') == torch.device('mps'):
                    torch.mps.empty_cache()
            # Check convergence (optional: compute full loss occasionally)
            if it % 100 == 0:
                print(f"Iteration {it}: Batch loss = {loss.item()}")
                if loss.item() < tol:
                    print(f"Converged at iteration {it} with batch loss = {loss.item()}")
                    break
    except KeyboardInterrupt:
        print("Interrupted by user.")

    del W_csr, N, M, d
    gc.collect()
    if torch.backends.mps.is_available() and torch.device('mps') == torch.device('mps'):
        torch.mps.empty_cache()
    return U, b


def orthogonalize(U):
    """
    Orthogonalize the matrix U to obtain principal components.
    Args:
        U: Matrix of shape (N, d) where d is the number of components.

    Returns:
        U_orth: Orthogonalized U of shape (N, d).
    """
    print("Orthogonalizing U...")
    Q, _ = torch.linalg.qr(U)  # QR decomposition for orthogonalization
    return Q

def kmeans_clustering(data, n_clusters, max_iter=100, tol=1e-4, device="cuda"):
    # Perform k-means clustering on the given data using PyTorch.
    n_samples, n_features = data.shape
    data = data.to(device)
    indices = torch.randint(0, n_samples, (n_clusters,), device=device)
    cluster_centers = data[indices]

    distances = torch.cdist(data, cluster_centers, p=2)
    labels = torch.argmin(distances, dim=1)

    for i in range(max_iter):
        distances = torch.cdist(data, cluster_centers, p=2)
        labels = torch.argmin(distances, dim=1)
        del distances
        new_cluster_centers = torch.stack([
            data[labels == k].mean(dim=0) if (labels == k).sum() > 0 else cluster_centers[k]
            for k in range(n_clusters)
        ])
        shift = torch.norm(new_cluster_centers - cluster_centers, p='fro').item()
        if shift < tol:
            print(f"K-means converged in {i + 1} iterations with shift={shift:.6f}")
            break
        cluster_centers = new_cluster_centers
        del new_cluster_centers
        if i % 10 == 0:
            gc.collect()
            if torch.backends.mps.is_available() and torch.device('mps') == torch.device('mps'):
                torch.mps.empty_cache()

    # Calculate distances to the assigned cluster centers for each example
    distances = torch.cdist(data, cluster_centers, p=2)
    min_distances = distances.gather(1, labels.unsqueeze(1)).squeeze()
    del n_samples, n_features, data, indices, distances
    gc.collect()
    if torch.backends.mps.is_available() and torch.device('mps') == torch.device('mps'):
        torch.mps.empty_cache()
    return cluster_centers, labels, min_distances

def run_pca_kmeans(batch, index, device="cuda"):
    print(f"[DEBUG] Using device: {device}")

    iteration = (batch * 4) + index
    # Perform stochastic PCA
    n_components = 61

# Best params: [61, 1278, 0.0452454837130382, 15000, "Adam"]

    # Perform k-means clustering
    n_clusters = 18  # Number of clusters

    # Load the sparse matrix
    file_path = "./worm_flywire_format_out/worm_herm_chemical_binary_connectivity.npz"
    adj_matrix = load_npz(file_path)
    print(f"Loaded sparse matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.")

    U, b = stochastic_pca(adj_matrix,
                      n_components,
                      batch_size=128,
                      lr=0.0452454837130382,
                    #   max_iter=15000,
                      max_iter=500,
                      tol=1e-6,
                      optimizer_choice="Adam",
                      device=device)

    # Orthogonalize U to obtain principal components
    U_orth = orthogonalize(U)
    # Run k-means once and return cluster centers.
    cluster_centers, labels, min_distances = kmeans_clustering(U_orth, n_clusters, device=device)

    print("Cluster centers shape:", cluster_centers.shape)
    print("Cluster labels shape:", labels.shape)
    print("Distances to cluster centers shape:", min_distances.shape)

    mapping = load_mapping('./worm_flywire_format_out/worm_neuron_to_index_mapping.json')
    rootid_mapping = dict((v, k) for k, v in mapping.items())

   # build a cluster assignment dictionary
    cluster_assignment_dict = dict()
    for i in range(len(labels)):
        root_id = mapping[i]
        cluster_assignment_dict[root_id] = labels[i].item()

    # Save the results
    torch.save(U_orth.cpu(), f"runs_worm/U_orth_tuned_{iteration}.pt")
    print("Saved U and U_orth to disk.")

    torch.save(cluster_centers.cpu(), f"runs_worm/pca_cluster_centers_tuned_{iteration}.pt")
    torch.save(labels.cpu(), f"runs_worm/pca_labels_tuned_{iteration}.pt")
    torch.save(min_distances.cpu(), f"runs_worm/pca_min_distances_tuned_{iteration}.pt")
    np.save(f"runs_worm/pca_cluster_assignment_dict_tuned_{iteration}.npy", cluster_assignment_dict, allow_pickle=True)


    del U, b, adj_matrix, U_orth, mapping, rootid_mapping, cluster_centers, labels, min_distances, cluster_assignment_dict
    gc.collect()
    if torch.backends.mps.is_available() and torch.device('mps') == torch.device('mps'):
        torch.mps.empty_cache()



def multi_run_kmeans_parallel_stability(n_batches=10, batch_size=5, device="cuda"):

    for batch in range(n_batches):
        _ = Parallel(n_jobs=-1)(
            delayed(run_pca_kmeans)(batch, index+1, device) for index in range(batch_size)
        )

        gc.collect()
    if torch.backends.mps.is_available() and torch.device('mps') == torch.device('mps'):
        torch.mps.empty_cache()



if __name__ == "__main__":
    n_batches = 12
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"


    multi_run_kmeans_parallel_stability(
        n_batches=n_batches, batch_size=batch_size, device=device
    )
