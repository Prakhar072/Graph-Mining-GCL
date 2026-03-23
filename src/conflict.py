"""
Conflict index computation measuring homophily-heterophily regime.

Computes the conflict between structural similarity (PPR) and attribute similarity
(cosine similarity) across sampled node pairs.
"""

import torch
import numpy as np
from pathlib import Path
import pickle


def compute_ppr_matrix(A_norm, num_nodes, alpha=0.15, num_iterations=10, device=None):
    """
    Compute Personalized PageRank (PPR) using power iteration.

    Uses the approximation: H = (1 - alpha) * A_norm @ H + alpha * I
    by iterating for num_iterations steps starting from H = I.

    Args:
        A_norm (torch.Tensor): Normalized adjacency matrix, shape (N, N)
                              Can be dense or sparse
        num_nodes (int): Number of nodes
        alpha (float): Teleport probability for PPR, default 0.15
        num_iterations (int): Number of power iterations
        device: Device for computation

    Returns:
        torch.Tensor: PPR matrix, shape (N, N), NOT row-normalized to preserve true values
    """
    if device is None:
        device = A_norm.device

    # Initialize H = I
    H = torch.eye(num_nodes, dtype=A_norm.dtype, device=device)

    # Handle sparse tensor
    #request change: if we are using dense vector anyway might as well compute it properly from the start instead of converting from sparse to dense
    is_sparse = A_norm.is_sparse
    if is_sparse:
        A_norm_dense = A_norm.coalesce().to_dense()
    else:
        A_norm_dense = A_norm

    # Power iteration
    for _ in range(num_iterations):
        H = (1 - alpha) * (A_norm_dense @ H) + alpha * torch.eye(num_nodes, dtype=H.dtype, device=device)

    # Do NOT row-normalize - we want to preserve raw PPR values for conflict detection
    return H


def compute_cosine_similarity_matrix(X):
    """
    Compute cosine similarity matrix for row-normalized features.

    Since X is already row-normalized, cosine similarity reduces to dot product.
    We keep values in [-1, 1] without row-normalization to preserve true similarity.

    Args:
        X (torch.Tensor): Row-normalized feature matrix, shape (N, D)

    Returns:
        torch.Tensor: Cosine similarity matrix, shape (N, N), values in [-1, 1]
    """
    # X is already row-normalized, so cosine sim = X @ X.T
    cos_sim = torch.mm(X, X.t())

    # Ensure values are in [-1, 1] (handle numerical errors)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # Do NOT row-normalize - we want to preserve raw cosine values for conflict detection
    return cos_sim


def compute_conflict_index(A, X, n_samples=10000, alpha=0.15, device=None):
    """
    Compute conflict index measuring divergence between structural and attribute similarity.

    Samples n_samples pairs: half from existing edges, half random non-edges.
    Computes structural similarity using PPR and attribute similarity using cosine.
    Conflict index C = mean |PPR(i,j) - cos_sim(i,j)| over sampled pairs.

    Args:
        A (torch.Tensor): Adjacency matrix (sparse or dense), shape (N, N)
        X (torch.Tensor): Row-normalized feature matrix, shape (N, D)
        n_samples (int): Number of pairs to sample, default 10000
        alpha (float): PPR teleport probability
        device: Computation device

    Returns:
        float: Conflict index C in [0, 1]
    """
    num_nodes = X.shape[0]

    if device is None:
        device = X.device

    print(f"Computing conflict index with {n_samples} samples...")

    # Convert adjacency to dense if sparse
    if A.is_sparse:
        A_dense = A.coalesce().to_dense()
    else:
        A_dense = A

    # Ensure A_dense has self-loops (should already have them from preprocessing)
    # Get list of existing edges
    edge_indices = torch.nonzero(A_dense, as_tuple=False)  # (num_edges, 2)
    num_edges = edge_indices.shape[0]

    # Compute PPR matrix
    ppr_matrix = compute_ppr_matrix(A_dense, num_nodes, alpha)

    # Compute cosine similarity matrix
    cos_sim_matrix = compute_cosine_similarity_matrix(X)

    # Sample pairs
    n_pos_samples = n_samples // 2
    n_neg_samples = n_samples // 2

    conflict_values = []

    # Sample from existing edges (positive pairs)
    if num_edges > 0:
        pos_indices = torch.randperm(num_edges, device='cpu')[:min(n_pos_samples, num_edges)]
        pos_pairs = edge_indices[pos_indices]

        for i, j in pos_pairs:
            ppr_sim = ppr_matrix[i, j].item()
            cos_sim = cos_sim_matrix[i, j].item()
            # Map PPR from [0, 1] to [-1, 1] to preserve negative information from cosine
            ppr_norm = ppr_sim * 2 - 1  # Map [0, 1] to [-1, 1]
            cos_norm = cos_sim  # Keep cosine as is, already in [-1, 1]
            conflict = abs(ppr_norm - cos_norm)
            conflict_values.append(conflict)

    # Sample random non-edges (negative pairs)
    neg_samples_collected = 0
    while neg_samples_collected < n_neg_samples:
        i = torch.randint(0, num_nodes, (1,)).item()
        j = torch.randint(0, num_nodes, (1,)).item()

        if i != j and A_dense[i, j].item() == 0:  # Non-edge
            ppr_sim = ppr_matrix[i, j].item()
            cos_sim = cos_sim_matrix[i, j].item()
            # Map PPR from [0, 1] to [-1, 1] to preserve negative information from cosine
            ppr_norm = ppr_sim * 2 - 1  # Map [0, 1] to [-1, 1]
            cos_norm = cos_sim  # Keep cosine as is, already in [-1, 1]
            conflict = abs(ppr_norm - cos_norm)
            conflict_values.append(conflict)
            neg_samples_collected += 1

    # Compute mean conflict
    C = np.mean(conflict_values) if conflict_values else 0.0

    print(f"  Conflict index C = {C:.6f}")

    return C


if __name__ == "__main__":
    print("Testing conflict index computation...\n")

    # Create test graph
    N = 100
    D = 50

    # Create random adjacency with some structure
    A = torch.rand(N, N) > 0.9
    A = A.float()
    A = A + A.t()  # Make symmetric
    torch.fill_diagonal(A, 1.0)  # Add self-loops

    # Create random features and normalize
    X = torch.randn(N, D)
    X_norm = X / (torch.norm(X, p=2, dim=1, keepdim=True) + 1e-8)

    print(f"Test graph: N={N}, D={D}, edges={A.sum().item()/2:.0f}")

    # Compute conflict index
    C = compute_conflict_index(A, X_norm, n_samples=1000, alpha=0.15)

    print(f"\nComputed conflict index: {C:.6f}")
    assert 0 <= C <= 1, f"Conflict index should be in [0, 1], got {C}"
    print("[PASS] Conflict index is valid")
