"""
Computation of structural and attribute weight matrices.

Generates supervision weight matrices W_s (structural) and W_a (attribute)
for soft contrastive learning.
"""

import torch
import numpy as np
from .conflict import compute_ppr_matrix, compute_cosine_similarity_matrix


def compute_structural_weights(A, n_nodes, method='ppr', alpha=0.15,
                               num_iterations=10, device=None):
    """
    Compute structural weight matrix using PPR similarity.

    Args:
        A (torch.Tensor): Adjacency matrix (normalized), shape (N, N)
                         Can be sparse or dense
        n_nodes (int): Number of nodes
        method (str): Method for structural similarity ('ppr' is default)
        alpha (float): PPR teleport probability
        num_iterations (int): PPR iterations
        device: Computation device

    Returns:
        torch.Tensor: Structural weight matrix W_s, shape (N, N), row-normalized
    """
    if device is None:
        device = A.device

    if method == 'ppr':
        # Use PPR for structural similarity
        W_s = compute_ppr_matrix(A, n_nodes, alpha, num_iterations, device)
    else:
        raise ValueError(f"Unknown method: {method}")

    # since ppr is computed without row normalziation, we need to row-normalize it here
    row_sums = W_s.sum(dim=1, keepdim=True)
    row_sums = torch.clamp(row_sums, min=1e-8)
    W_s = W_s / row_sums

    return W_s


def compute_attribute_weights(X):
    """
    Compute attribute weight matrix using cosine similarity on row-normalized features.

    Args:
        X (torch.Tensor): Row-normalized feature matrix, shape (N, D)

    Returns:
        torch.Tensor: Attribute weight matrix W_a, shape (N, N), row-normalized
    """
    # Compute cosine similarity using the dedicated function
    W_a = compute_cosine_similarity_matrix(X)

    # Make sure values are non-negative by taking ReLU
    W_a = torch.relu(W_a)

    # Row-normalize with a robust fallback:
    # if a row is all zeros (possible on heterophilic/sparse-feature nodes),
    # assign a self-weight of 1 so every row is a valid probability distribution.
    row_sums = W_a.sum(dim=1, keepdim=True)
    zero_rows = row_sums.squeeze(1) <= 1e-12
    if zero_rows.any():
        idx = torch.nonzero(zero_rows, as_tuple=False).squeeze(1)
        W_a[idx, idx] = 1.0
        row_sums = W_a.sum(dim=1, keepdim=True)

    row_sums = torch.clamp(row_sums, min=1e-8)
    W_a = W_a / row_sums

    return W_a


if __name__ == "__main__":
    print("Testing weight matrix computation...\n")

    N = 100
    D = 50

    # Create test adjacency (normalized)
    A = torch.rand(N, N) > 0.9
    A = A.float()
    A = A + A.t()
    torch.fill_diagonal(A, 1.0)
    # Normalize: D^{-1/2} * A * D^{-1/2}
    degree = A.sum(dim=1)
    deg_inv_sqrt = torch.pow(degree, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    A_norm = deg_inv_sqrt.unsqueeze(1) * A * deg_inv_sqrt.unsqueeze(0)

    # Create test features
    X = torch.randn(N, D)
    X_norm = X / (torch.norm(X, p=2, dim=1, keepdim=True) + 1e-8)

    print(f"Test data: N={N}, D={D}")

    # Test structural weights
    print("\nTesting structural weights:")
    W_s = compute_structural_weights(A_norm, N, method='ppr', alpha=0.15)
    print(f"  Shape: {W_s.shape}")
    assert W_s.shape == (N, N), f"Expected ({N}, {N}), got {W_s.shape}"
    row_sums = W_s.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(N), atol=1e-5), "Rows should sum to 1"
    print("  [PASS] Row-normalized")
    print(f"  Mean weight: {W_s.mean().item():.6f}")

    # Test attribute weights
    print("\nTesting attribute weights:")
    W_a = compute_attribute_weights(X_norm)
    print(f"  Shape: {W_a.shape}")
    assert W_a.shape == (N, N), f"Expected ({N}, {N}), got {W_a.shape}"
    row_sums = W_a.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(N), atol=1e-5), "Rows should sum to 1"
    print("  [PASS] Row-normalized")
    print(f"  Mean weight: {W_a.mean().item():.6f}")

    # Test slicing
    print("\nTesting weight slicing:")
    batch_size = 10
    batch_indices = torch.randint(0, N, (batch_size,))
    W_s_batch = slice_weight_matrix(W_s, batch_indices)
    print(f"  Sliced shape: {W_s_batch.shape}")
    assert W_s_batch.shape == (batch_size, batch_size), f"Expected ({batch_size}, {batch_size})"
    row_sums = W_s_batch.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(batch_size), atol=1e-5), "Batch rows should sum to 1"
    print("  [PASS] Batch slicing works")

    print("\nAll weight tests passed!")
