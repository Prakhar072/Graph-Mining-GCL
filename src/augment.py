"""
Graph augmentation functions for contrastive learning.

Implements edge dropping (via Bernoulli sampling) and feature masking
for creating augmented views.
"""

import torch
import torch_geometric


def drop_edges(edge_index, num_nodes, p_e):
    """
    Drop edges from the graph using Bernoulli sampling.

    Args:
        edge_index (torch.Tensor): Edge indices, shape (2, num_edges)
        num_nodes (int): Number of nodes
        p_e (float): Edge drop probability, in [0, 1]

    Returns:
        torch.Tensor: New edge_index with dropped edges removed, shape (2, num_remaining_edges)
    """
    if p_e < 0 or p_e > 1:
        raise ValueError(f"Drop probability p_e must be in [0, 1], got {p_e}")

    if p_e == 0:
        return edge_index

    num_edges = edge_index.shape[1]

    # Create mask for edges to keep (1 - p_e probability)
    # Each edge is kept with probability (1 - p_e)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > p_e

    # Apply mask to edge_index
    edge_index_dropped = edge_index[:, keep_mask]

    return edge_index_dropped


def mask_features(X, p_f):
    """
    Mask features by zeroing dimensions independently with probability p_f.

    Args:
        X (torch.Tensor): Feature matrix, shape (N, D)
        p_f (float): Feature mask probability, in [0, 1]

    Returns:
        torch.Tensor: Masked feature matrix, shape (N, D)
    """
    if p_f < 0 or p_f > 1:
        raise ValueError(f"Mask probability p_f must be in [0, 1], got {p_f}")

    if p_f == 0:
        return X

    N, D = X.shape

    # Create mask for features to keep (1 - p_f probability)
    # Each feature dimension is kept with probability (1 - p_f)
    keep_mask = torch.rand(D, device=X.device) > p_f

    # Apply mask to features - this creates a new tensor, not in-place
    X_masked = X * keep_mask.float()

    return X_masked


def augment_graph(edge_index, X, p_e, p_f):
    """
    Apply both edge dropping and feature masking to create an augmented view.

    Args:
        edge_index (torch.Tensor): Edge indices, shape (2, num_edges)
        X (torch.Tensor): Feature matrix, shape (N, D)
        p_e (float): Edge drop probability
        p_f (float): Feature mask probability

    Returns:
        tuple: (edge_index_aug, X_aug)
    """
    edge_index_aug = drop_edges(edge_index, X.shape[0], p_e)
    X_aug = mask_features(X, p_f)

    return edge_index_aug, X_aug


if __name__ == "__main__":
    # Test augmentation functions
    print("Testing augmentation functions...\n")

    # Create test data
    N, D = 100, 50
    num_edges = 500

    X = torch.randn(N, D)
    edge_index = torch.randint(0, N, (2, num_edges))

    print(f"Original: N={N}, D={D}, num_edges={num_edges}")

    # Test edge dropping
    print("\nTesting edge dropping:")
    for p_e in [0.0, 0.1, 0.3, 0.5]:
        edge_index_dropped = drop_edges(edge_index, N, p_e)
        print(f"  p_e={p_e}: {edge_index.shape[1]} -> {edge_index_dropped.shape[1]} edges")

    # Test feature masking
    print("\nTesting feature masking:")
    for p_f in [0.0, 0.1, 0.3, 0.5]:
        X_masked = mask_features(X, p_f)
        zeros_count = (X_masked == 0).sum().item()
        total = X_masked.numel()
        print(f"  p_f={p_f}: {zeros_count}/{total} elements zeroed (~{zeros_count/total:.1%})")

    # Test combined augmentation
    print("\nTesting combined augmentation:")
    for p_e, p_f in [(0.3, 0.3), (0.5, 0.2), (0.2, 0.5)]:
        edge_index_aug, X_aug = augment_graph(edge_index, X, p_e, p_f)
        print(f"  p_e={p_e}, p_f={p_f}: edges {num_edges}->{edge_index_aug.shape[1]}, " +
              f"masked features ~{(X_aug==0).sum().item()/X_aug.numel():.1%}")

    print("\nAll tests passed!")
