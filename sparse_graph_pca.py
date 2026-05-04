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

space = [
    Integer(24, 40, name='n_components'),
    Integer(256, 1024, name='k'),
    Real(1e-4, 1e-1, prior='log-uniform', name='learning_rate'),
    Integer(5000, 15000, name='n_epochs'),
    Categorical(['Adam', 'AdamW', 'SGD'], name='optimizer')  # Optimizer choice
]



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
            torch.mps.empty_cache()

    # Calculate distances to the assigned cluster centers for each example
    distances = torch.cdist(data, cluster_centers, p=2)
    min_distances = distances.gather(1, labels.unsqueeze(1)).squeeze()
    del n_samples, n_features, data, indices, distances
    gc.collect()
    torch.mps.empty_cache()
    return cluster_centers, labels, min_distances

@use_named_args(space)
def objective(**params):
   # run = wdb.init(
   #     project="flywrite",
   # )

    n_components = params["n_components"]
    n_clusters = params["k"]
    learning_rate = params["learning_rate"]
    n_epochs = params["n_epochs"]
    optimizer_choice = params["optimizer"]

    # Load the sparse matrix
    file_path = "./sparse_connectivity_matrix.npz"
    adj_matrix = load_npz(file_path)
    print(f"Loaded sparse matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.")

    U, b = stochastic_pca(adj_matrix, 
                      n_components, 
                      batch_size=512, 
                      lr=learning_rate, 
                      max_iter=n_epochs,
                      tol=1e-6,
                      optimizer_choice=optimizer_choice,
                      device=device)
    
    # with torch.no_grad():
    # Orthogonalize U to obtain principal components
    U_orth = orthogonalize(U)

    cluster_centers, labels, min_distances = kmeans_clustering(U_orth, n_clusters, device=device)

    print("Cluster centers shape:", cluster_centers.shape)
    print("Cluster labels shape:", labels.shape)
    print("Distances to cluster centers shape:", min_distances.shape)

    del cluster_centers, min_distances, U, b, adj_matrix

    # Compute the silhouette score.
    default_evaluator = Engine(eval_step)
    # score = silhouette_score(U_orth.detach().cpu().numpy(), labels.detach().cpu().numpy())
    metric = SilhouetteScore()
    metric.attach(default_evaluator, "silhouette_score")
    state = default_evaluator.run([{"features": U_orth, "labels": labels}])
    score = state.metrics["silhouette_score"]
    
    del labels, U_orth, metric, default_evaluator, state
    gc.collect()
    torch.mps.empty_cache()

    return -score
    # Since gp_minimize minimizes the objective, return the negative silhouette score.
    # return -score
    # avg_min_distance = min_distances.mean().item()

    # return avg_min_distance

def eval_step(engine, batch):
    return batch

def get_lr(optimizer):
    result = []
    for param_group in optimizer.param_groups:
        result.append(param_group['lr'])
    return sum(result) / len(result)

def run_pca_kmeans(batch, index, device="cuda"):
    iteration = (batch * 4) + index
    # Perform stochastic PCA
    n_components = 24

    # Perform k-means clustering
    n_clusters = 1024  # Number of clusters

    # Load the sparse matrix
    file_path = "./sparse_connectivity_matrix.npz"
    adj_matrix = load_npz(file_path)
    print(f"Loaded sparse matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.")

    U, b = stochastic_pca(adj_matrix, 
                      n_components, 
                      batch_size=512, 
                      lr=0.1, 
                      max_iter=15000,
                      tol=1e-6,
                      optimizer_choice="SGD",
                      device=device)
    
    # Orthogonalize U to obtain principal components
    U_orth = orthogonalize(U)
    # Run k-means once and return cluster centers.
    cluster_centers, labels, min_distances = kmeans_clustering(U_orth, n_clusters, device=device)

    print("Cluster centers shape:", cluster_centers.shape)
    print("Cluster labels shape:", labels.shape)
    print("Distances to cluster centers shape:", min_distances.shape)

    mapping = load_mapping('./root_id_to_index_mapping.json')
    rootid_mapping = dict((v, k) for k, v in mapping.items())

   # build a cluster assignment dictionary
    cluster_assignment_dict = dict()
    for i in range(len(labels)):
        root_id = mapping[i]
        cluster_assignment_dict[root_id] = labels[i].item()

    # Save the results
    torch.save(U_orth.cpu(), f"runs/U_orth_tuned_{iteration}.pt")
    print("Saved U and U_orth to disk.")

    torch.save(cluster_centers.cpu(), f"runs/pca_cluster_centers_tuned_{iteration}.pt")
    torch.save(labels.cpu(), f"runs/pca_labels_tuned_{iteration}.pt")
    torch.save(min_distances.cpu(), f"runs/pca_min_distances_tuned_{iteration}.pt")
    np.save(f"runs/pca_cluster_assignment_dict_tuned_{iteration}.npy", cluster_assignment_dict, allow_pickle=True)

    del U, b, adj_matrix, U_orth, mapping, rootid_mapping, cluster_centers, labels, min_distances, cluster_assignment_dict
    gc.collect()
    torch.mps.empty_cache()

#     return centers

def align_centers(reference, centers):
    cost_matrix = torch.cdist(reference, centers, p=2).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    aligned = centers[col_ind]
    return aligned

def compute_intervals(aligned_results):
    # aligned_results is (n_runs, n_clusters, n_features)
    mean_centers = np.mean(aligned_results, axis=0)
    lower_bounds = np.percentile(aligned_results, 2.5, axis=0)
    upper_bounds = np.percentile(aligned_results, 97.5, axis=0)
    return mean_centers, lower_bounds, upper_bounds

def multi_run_kmeans_parallel_stability(n_batches=10, batch_size=5, device="cuda"):
#    aligned_results_all = []
#    cumulative_means, cumulative_lower, cumulative_upper = [], [], []
    
    # Run k-means in batches until we have n_runs total
    for batch in range(n_batches):
        _ = Parallel(n_jobs=-1)(
            delayed(run_pca_kmeans)(batch, index+1, device) for index in range(batch_size)
        )
#        # Align each run in the batch to the reference from the first run of the entire process
#        if batch == 0:
#            reference = batch_results[0]
#        for centers in batch_results:
#            aligned = align_centers(reference, centers)
#            aligned_results_all.append(aligned.detach().cpu().numpy())
#            del aligned
#        
#        # Compute cumulative intervals after this batch
#        aligned_array = np.stack(aligned_results_all, axis=0)  # shape: (runs_so_far, n_clusters, n_features)
#        mean_centers, lower_bounds, upper_bounds = compute_intervals(aligned_array)
#        cumulative_means.append(mean_centers)
#        cumulative_lower.append(lower_bounds)
#        cumulative_upper.append(upper_bounds)
#        
#        print(f"After {(batch+1)*batch_size} runs, cumulative mean for cluster 0: {mean_centers[0]}")
#        del mean_centers, lower_bounds, upper_bounds, batch_results, aligned_array
        gc.collect()
        torch.mps.empty_cache()
        
#     return cumulative_means, cumulative_lower, cumulative_upper

if __name__ == "__main__":
    # Set device
    device="mps"
    # if torch.backends.mps.is_available():
    #     device="mps"
    if torch.cuda.is_available():
        device="cuda"

    # res_gp = gp_minimize(objective, space, acq_func="EI", n_calls=50, n_jobs=10, random_state=42)

    n_batches = 12
    batch_size = 4 

#    opt = Optimizer(dimensions=space, base_estimator="GP", acq_func="EI", random_state=42)
#
#    for i in range(n_batches):
#        candidates = opt.ask(n_points=batch_size)
#
#        if candidates is None:
#            raise ValueError("opt.ask() returned None")
#
#        # with mp.Pool(processes=batch_size) as pool:
#        #     scores = pool.map(run_evaluation, candidates)
#
#        with Parallel(n_jobs=batch_size) as parallel:
#            scores = parallel(delayed(objective)(params) for params in candidates)
#        # scores = Parallel(n_jobs=batch_size)(
#        #     delayed(objective)(params) for params in candidates
#        # )
#        
#        opt.tell(candidates, scores)
#        print(f"Batch {i+1}: Best score so far = {-min(opt.yi):.4f}")
#
#    # Best config
#    best_idx = np.argmin(opt.yi)
#    print("\nBest configuration:")
#    print(f"  Params: {opt.Xi[best_idx]}")
#    print(f"  Silhouette Score: {-opt.yi[best_idx]:.4f}")

#    # Print the results.
#    print("Best min distance: {:.4f}".format(None if res_gp is None else -res_gp.fun))
#    print("Best hyperparameters:")
#    print("  n_components:", None if res_gp is None else res_gp.x[0])
#    print("  k (clusters):", None if res_gp is None else res_gp.x[1])
#    print("  learning_rate:", None if res_gp is None else res_gp.x[2])
#    print("  n_epochs:", None if res_gp is None else res_gp.x[3])
#    print("  optimizer:", None if res_gp is None else res_gp.x[4])

    # Implement credible intervals

    multi_run_kmeans_parallel_stability(
        n_batches=n_batches, batch_size=batch_size, device=device
    )

#    np.save("kmeans_mean_centers.npy", cum_means)
#    np.save("kmeans_lower_bounds.npy", cum_lower)
#    np.save("kmeans_upper_bounds.npy", cum_upper)

#    n_components = 24
#    n_clusters = 1024
#    learning_rate = 0.1
#    n_epochs = 15000
#    optimizer_choice = "SGD"
#
#    # Load the sparse matrix
#    file_path = "./sparse_connectivity_matrix.npz"
#    adj_matrix = load_npz(file_path)
#    print(f"Loaded sparse matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.")
#
#    U, b = stochastic_pca(adj_matrix, 
#                      n_components, 
#                      batch_size=512, 
#                      lr=learning_rate, 
#                      max_iter=n_epochs,
#                      tol=1e-6,
#                      optimizer_choice=optimizer_choice,
#                      device=device)
#    
#    # with torch.no_grad():
#    # Orthogonalize U to obtain principal components
#    U_orth = orthogonalize(U)
#
#    cluster_centers, labels, min_distances = kmeans_clustering(U_orth, n_clusters, device=device)
#
#    print("Cluster centers shape:", cluster_centers.shape)
#    print("Cluster labels shape:", labels.shape)
#    print("Distances to cluster centers shape:", min_distances.shape)
#
#    # build a index-to-root_id dictionary
#    from index_mapping import load_mapping
#
#    mapping = load_mapping('./root_id_to_index_mapping.json')
#    rootid_mapping = dict((v, k) for k, v in mapping.items())
#
#   # build a cluster assignment dictionary
#    cluster_assignment_dict = dict()
#    for i in range(len(labels)):
#        root_id = mapping[i]
#        cluster_assignment_dict[root_id] = labels[i].item()
#
#    # Save the results
#    torch.save(U_orth.cpu(), 'U_orth_tuned.pt')
#    print("Saved U and U_orth to disk.")
#
#    torch.save(cluster_centers.cpu(), 'pca_cluster_centers_tuned.pt')
#    torch.save(labels.cpu(), 'pca_labels_tuned.pt')
#    torch.save(min_distances.cpu(), 'pca_min_distances_tuned.pt')
#    np.save("pca_cluster_assignment_dict_tuned.npy", cluster_assignment_dict, allow_pickle=True)
