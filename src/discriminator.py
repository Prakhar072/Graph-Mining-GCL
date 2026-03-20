"""
Semantics-consistency discriminator for pair evaluation.

Implements a discriminator MLP that evaluates semantic consistency of node pairs,
used to calibrate weights during fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors


class Discriminator(nn.Module):
    """
    Three-layer MLP discriminator for pair evaluation.

    Args:
        in_dim (int): Input dimension (d + t)
        hidden_dim (int): First hidden layer dimension
    """

    def __init__(self, in_dim, hidden_dim):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)


def get_eigenvectors(A_norm, t=64, device=None):
    """Get top-t eigenvectors of normalized Laplacian."""
    if device is None:
        device = A_norm.device

    N = A_norm.shape[0]

    if A_norm.is_sparse:
        I = torch.eye(N, dtype=A_norm.dtype, device=device)
        L = I - A_norm.to_dense()
    else:
        L = torch.eye(N, dtype=A_norm.dtype, device=device) - A_norm

    if N <= 3000:
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        top_indices = torch.argsort(eigenvalues)[:t]
        eigenvectors_top = eigenvectors[:, top_indices]
    else:
        try:
            from scipy.sparse.linalg import eigsh
            import scipy.sparse as sp

            if A_norm.is_sparse:
                A_scipy = sp.coo_matrix(
                    (A_norm.coalesce().values().cpu().numpy(),
                     A_norm.coalesce().indices().cpu().numpy()),
                    shape=(N, N)
                )
            else:
                A_scipy = sp.csr_matrix(A_norm.cpu().numpy())

            I_scipy = sp.eye(N)
            L_scipy = I_scipy - A_scipy

            eigenvalues, eigenvectors = eigsh(L_scipy, k=t, which='SM')
            eigenvectors_top = torch.from_numpy(eigenvectors.astype(np.float32)).to(device)

        except ImportError:
            print("Warning: scipy not available, falling back to dense")
            L_dense = L.cpu().numpy()
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            top_indices = np.argsort(eigenvalues)[:t]
            eigenvectors_top = torch.from_numpy(
                eigenvectors[:, top_indices].astype(np.float32)
            ).to(device)

    return eigenvectors_top


def build_fusion_vectors(H, C_struct):
    """Build fusion vectors by concatenating H and structural eigenvectors."""
    Z = torch.cat([H, C_struct], dim=1)
    return Z


def get_pair_scores(discriminator, Z, pair_indices):
    """Get discriminator scores for node pairs using Hadamard product."""
    i_indices = pair_indices[:, 0]
    j_indices = pair_indices[:, 1]

    Z_i = Z[i_indices]
    Z_j = Z[j_indices]

    Z_combined = Z_i * Z_j

    scores = discriminator(Z_combined)

    return scores


def build_pretrain_pairs(A, X, k=10):
    """Build positive and negative pairs for pre-training discriminator."""
    N = A.shape[0]
    device = A.device

    if A.is_sparse:
        A_dense = A.coalesce().to_dense()
    else:
        A_dense = A

    A_no_diag = A_dense.clone()
    A_no_diag.diagonal().zero_()

    X_np = X.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X_np)
    _, knn_indices = nbrs.kneighbors(X_np)

    knn_indices = torch.from_numpy(knn_indices).to(device)

    pos_pairs = []
    for i in range(N):
        neighbors = torch.nonzero(A_no_diag[i], as_tuple=False).squeeze(-1)
        knn_set = set(knn_indices[i, 1:].tolist())

        for j in neighbors:
            if j.item() in knn_set:
                pos_pairs.append([i, j.item()])

    pos_pairs = torch.tensor(pos_pairs, dtype=torch.long, device=device)

    neg_pairs = []
    neg_target = max(len(pos_pairs), 1)

    while len(neg_pairs) < neg_target:
        i = torch.randint(0, N, (1,)).item()
        j = torch.randint(0, N, (1,)).item()

        if i != j and A_dense[i, j].item() == 0:
            neg_pairs.append([i, j])

    neg_pairs = torch.tensor(neg_pairs, dtype=torch.long, device=device)

    return pos_pairs, neg_pairs


def build_finetune_pairs(A, H, k=10):
    """Build pairs for fine-tuning discriminator."""
    return build_pretrain_pairs(A, H, k=k)


def balanced_softmax_loss(scores, labels, n_pos, n_neg):
    """Compute balanced softmax loss for pair classification."""
    device = scores.device

    adjustment = torch.where(
        labels == 1,
        torch.tensor(np.log(n_pos + 1e-8), dtype=scores.dtype, device=device),
        torch.tensor(np.log(n_neg + 1e-8), dtype=scores.dtype, device=device)
    )

    adjusted_scores = scores + adjustment
    loss = F.binary_cross_entropy_with_logits(adjusted_scores, labels.float())

    return loss
