"""
Dataset loading and preprocessing for graph contrastive learning.

Loads graph datasets from PyTorch Geometric, normalizes adjacency matrices,
and row-normalizes features. Caches preprocessed data to disk.
"""

import os
import pickle
import torch
import numpy as np
from pathlib import Path
import torch_geometric
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor
import torch_geometric.transforms as T


def get_cache_dir():
    """Get or create cache directory for preprocessed data."""
    cache_dir = Path(__file__).parent.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def load_dataset(name, normalize_adj=True, normalize_features=True):
    """
    Load and preprocess a graph dataset.

    Args:
        name (str): Dataset name ('cora', 'citeseer', 'pubmed', 'chameleon',
                   'squirrel', 'actor')
        normalize_adj (bool): Whether to normalize adjacency matrix
        normalize_features (bool): Whether to row-normalize features

    Returns:
        tuple: (A_normed, X_normed, y, edge_index, N, D)
            - A_normed: Normalized adjacency matrix (sparse tensor or None)
            - X_normed: Row-normalized feature matrix, shape (N, D)
            - y: Node labels, shape (N,)
            - edge_index: Edge indices, shape (2, num_edges)
            - N: Number of nodes
            - D: Feature dimension
    """
    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{name}_preprocessed.pkl"

    # Try to load from cache
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Loading dataset: {name}")

    # Determine dataset type and load
    name_lower = name.lower()

    if name_lower in ['cora', 'citeseer', 'pubmed']:
        # Planetoid datasets (homophilic)
        dataset = Planetoid(
            root=str(cache_dir / 'raw'),
            name=name_lower.capitalize(),
            split='public'
        )
    elif name_lower in ['chameleon', 'squirrel']:
        # Wikipedia network datasets (heterophilic)
        dataset = WikipediaNetwork(
            root=str(cache_dir / 'raw'),
            name=name_lower
        )
    elif name_lower == 'actor':
        # Actor dataset (heterophilic)
        dataset = Actor(root=str(cache_dir / 'raw'))
    else:
        raise ValueError(f"Unknown dataset: {name}")

    data = dataset[0]

    # Extract basic information
    N = data.x.shape[0]
    D = data.x.shape[1]
    edge_index = data.edge_index
    y = data.y if hasattr(data, 'y') else torch.zeros(N, dtype=torch.long)

    # Get feature matrix
    X = data.x.float()

    # Row-normalize features
    if normalize_features:
        X_norm = torch.norm(X, p=2, dim=1, keepdim=True)
        X_norm = torch.clamp(X_norm, min=1e-8)  # Avoid division by zero
        X = X / X_norm

    # Create normalized adjacency matrix
    A_normed = None
    if normalize_adj:
        # Add self-loops to adjacency
        edge_index_with_loops = torch_geometric.utils.add_self_loops(edge_index, num_nodes=N)[0]

        # Compute degree
        degree = torch_geometric.utils.degree(
            edge_index_with_loops[0],
            num_nodes=N,
            dtype=X.dtype
        )

        # Compute D^{-1/2}
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        # Normalize: D^{-1/2} * A * D^{-1/2}
        edge_weight = deg_inv_sqrt[edge_index_with_loops[0]] * deg_inv_sqrt[edge_index_with_loops[1]]

        # Create sparse tensor for normalized adjacency
        A_normed = torch.sparse_coo_tensor(
            edge_index_with_loops,
            edge_weight,
            size=(N, N),
            dtype=X.dtype
        )

    result = (A_normed, X, y, edge_index, N, D)

    # Cache the result
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"Cached preprocessed data to {cache_file}")

    return result


def get_adjacency_dense(edge_index, num_nodes, dtype=torch.float):
    """
    Convert edge_index to dense adjacency matrix with self-loops.

    Args:
        edge_index (torch.Tensor): Edge indices, shape (2, num_edges)
        num_nodes (int): Number of nodes
        dtype: Data type for the output

    Returns:
        torch.Tensor: Dense adjacency matrix, shape (num_nodes, num_nodes)
    """
    # Add self-loops
    edge_index_with_loops = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)[0]

    # Create adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes, dtype=dtype)
    adj[edge_index_with_loops[0], edge_index_with_loops[1]] = 1.0

    return adj


def get_normalized_adjacency_dense(edge_index, num_nodes, dtype=torch.float):
    """
    Get normalized adjacency matrix D^{-1/2} * (A + I) * D^{-1/2}.

    Args:
        edge_index (torch.Tensor): Edge indices
        num_nodes (int): Number of nodes
        dtype: Data type for output

    Returns:
        torch.Tensor: Normalized adjacency, shape (num_nodes, num_nodes)
    """
    # Add self-loops
    edge_index_with_loops = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)[0]

    # Compute degree
    degree = torch_geometric.utils.degree(
        edge_index_with_loops[0],
        num_nodes=num_nodes,
        dtype=dtype
    )

    # Compute D^{-1/2}
    deg_inv_sqrt = torch.pow(degree, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Create normalized adjacency
    edge_weight = deg_inv_sqrt[edge_index_with_loops[0]] * deg_inv_sqrt[edge_index_with_loops[1]]

    adj_norm = torch.zeros(num_nodes, num_nodes, dtype=dtype)
    adj_norm[edge_index_with_loops[0], edge_index_with_loops[1]] = edge_weight

    return adj_norm


if __name__ == "__main__":
    # Test dataset loading
    datasets = ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor']

    for dataset_name in datasets:
        try:
            print(f"\nLoading {dataset_name}...")
            A_norm, X, y, edge_index, N, D = load_dataset(dataset_name)

            print(f"  Nodes: {N}, Features: {D}, Edges: {edge_index.shape[1]}")
            print(f"  Features normalized: X.min()={X.min():.4f}, X.max()={X.max():.4f}, X_norm={X.norm(dim=1).mean():.4f}")
            print(f"  Labels unique classes: {y.unique().shape[0]}")

        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}")
