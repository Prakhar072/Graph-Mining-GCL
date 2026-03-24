"""
Soft Contrastive Loss for weighted supervision.

This module implements the soft contrastive loss that uses adaptive weights
to balance structural and attribute-based supervision signals based on the
graph's conflict index (homophily-heterophily regime).
"""

import torch
import torch.nn.functional as F
import numpy as np

# Try relative import if available, otherwise absolute
#request change: should be able to use proper import now that we are in the same package
try:
    from .spart import spart_similarity_logits
except ImportError:
    from spart import spart_similarity_logits

# request rewrites: we are implementing something completely different from the outlined methodology


def balanced_softmax_loss(scores, labels, n_pos, n_neg):
    """
    Compute balanced softmax loss for pair classification.

    Args:
        scores (torch.Tensor): Discriminator scores for pairs
        labels (torch.Tensor): Binary labels (1 for positive, 0 for negative)
        n_pos (int): Number of positive pairs
        n_neg (int): Number of negative pairs

    Returns:
        torch.Tensor: Scalar loss value
    """
    device = scores.device

    adjustment = torch.where(
        labels == 1,
        torch.tensor(np.log(n_pos + 1e-8), dtype=scores.dtype, device=device),
        torch.tensor(np.log(n_neg + 1e-8), dtype=scores.dtype, device=device)
    )

    adjusted_scores = scores + adjustment
    loss = F.binary_cross_entropy_with_logits(adjusted_scores, labels.float())

    return loss


def soft_contrastive_loss(H_u, H_v, W, tau, k):
    """
    Compute soft contrastive loss with weighted supervision.

    This implements: L = -mean_i( sum_j W[i,j] * log_softmax(S[i,j]) )
    where S is the SPART similarity matrix and W provides soft labels
    that weight how much node j should be pulled toward node i.

    Args:
        H_u (torch.Tensor): Batch of representations from view u, shape (B, d)
        H_v (torch.Tensor): Batch of representations from view v, shape (B, d)
        W (torch.Tensor): Soft supervision weights, shape (B, B).
                         Each row must sum to 1.
        tau (float): Temperature parameter
        k (int): Number of partitions for SPART

    Returns:
        torch.Tensor: Scalar loss value
    """
    B, d = H_u.shape
    assert H_v.shape == (B, d), f"H_v must have shape ({B}, {d}), got {H_v.shape}"
    assert W.shape == (B, B), f"W must have shape ({B}, {B}), got {W.shape}"

    # Cross-view similarity: queries from u against keys from v
    S_uv = spart_similarity_logits(H_u, H_v, k, tau)  # shape: (B, B)
    # Symmetric direction: queries from v against keys from u
    S_vu = spart_similarity_logits(H_v, H_u, k, tau)  # shape: (B, B)

    # Compute log-softmax along dimension 1 (for each query)
    log_probs_uv = F.log_softmax(S_uv, dim=1)  # shape: (B, B)
    log_probs_vu = F.log_softmax(S_vu, dim=1)  # shape: (B, B)

    # Compute weighted loss for both directions and average.
    # For v->u, weights are transposed to preserve (query, key) semantics.
    loss_uv = -(W * log_probs_uv).sum() / B
    loss_vu = -(W.t() * log_probs_vu).sum() / B
    loss = 0.5 * (loss_uv + loss_vu)

    return loss


def compute_combined_weights(W_s, W_a, C, alpha, beta):
    """
    Combine structural and attribute weights based on conflict index.

    Uses a sigmoid to adaptively weight between structural (W_s) and attribute
    (W_a) supervision signals based on the conflict index C, which measures
    the homophily-heterophily regime of the graph.

    Args:
        W_s (torch.Tensor): Structural weight matrix, shape (N, N),
                           row-normalized (each row sums to 1)
        W_a (torch.Tensor): Attribute weight matrix, shape (N, N),
                           row-normalized (each row sums to 1)
        C (float): Conflict index, typically in [0, 1]
        alpha (float): Sigmoid sharpness parameter (controls steepness)
        beta (float): Sigmoid midpoint parameter (controls threshold)

    Returns:
        torch.Tensor: Combined weight matrix W_total, shape (N, N),
                     row-normalized
    """
    N = W_s.shape[0]
    assert W_s.shape == (N, N), f"W_s must be square, got {W_s.shape}"
    assert W_a.shape == (N, N), f"W_a must be square, got {W_a.shape}"

    # Verify row-normalization
    row_sums_s = W_s.sum(dim=1)
    row_sums_a = W_a.sum(dim=1)
    assert torch.allclose(row_sums_s, torch.ones(N, device=W_s.device, dtype=W_s.dtype), atol=1e-5), \
        "W_s rows must sum to 1"
    assert torch.allclose(row_sums_a, torch.ones(N, device=W_a.device, dtype=W_a.dtype), atol=1e-5), \
        "W_a rows must sum to 1"

    # Compute adaptive weight lambda_C using sigmoid
    # lambda_C = sigmoid(alpha * (C - beta))
    # When C > beta: lambda_C increases (more weight on W_a, attribute-based)
    # When C < beta: lambda_C decreases (more weight on W_s, structural)
    lambda_C = torch.sigmoid(torch.tensor(alpha * (C - beta), dtype=W_s.dtype, device=W_s.device))

    # Combine weights
    W_total = (1 - lambda_C) * W_s + lambda_C * W_a

    # Row-normalize the combined matrix
    row_sums = W_total.sum(dim=1, keepdim=True)
    row_sums = torch.clamp(row_sums, min=1e-8)  # Avoid division by zero
    W_total = W_total / row_sums

    return W_total


def _test_soft_contrastive_loss():
    """
    Test soft contrastive loss computation.
    """
    B, d = 16, 128
    k = 4  # 4 partitions of size 32
    tau = 0.8

    # Create test data
    H_u = torch.randn(B, d, requires_grad=True)
    H_v = torch.randn(B, d, requires_grad=True)

    # Create uniform weights (all nodes are equally weighted)
    W_uniform = torch.ones(B, B, device=H_u.device, dtype=H_u.dtype) / B

    # Compute loss
    loss = soft_contrastive_loss(H_u, H_v, W_uniform, tau, k)

    # Verify loss is scalar and finite
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.shape == torch.Size([]), "Loss should be scalar"

    # Test that loss is differentiable
    loss.backward()
    assert H_u.grad is not None, "Gradient should be computed for H_u"
    assert H_v.grad is not None, "Gradient should be computed for H_v"
    assert H_u.grad.shape == H_u.shape, "Gradient shape should match H_u shape"
    assert H_v.grad.shape == H_v.shape, "Gradient shape should match H_v shape"

    print("[PASS] Loss computation passed")
    print(f"[PASS] Loss value: {loss.item():.6f}")
    print(f"[PASS] Gradient shapes: H_u={H_u.grad.shape}, H_v={H_v.grad.shape}")

    return True


def _test_combined_weights():
    """
    Test combined weight computation.
    """
    N = 64

    # Create test matrices (row-normalized)
    W_s = torch.rand(N, N)
    W_s = W_s / W_s.sum(dim=1, keepdim=True)

    W_a = torch.rand(N, N)
    W_a = W_a / W_a.sum(dim=1, keepdim=True)

    # Test with different conflict indices
    test_cases = [
        (0.2, "low conflict (homophilic)"),
        (0.5, "medium conflict"),
        (0.8, "high conflict (heterophilic)"),
    ]

    for C, description in test_cases:
        W_total = compute_combined_weights(W_s, W_a, C, alpha=5.0, beta=0.5)

        # Verify output shape
        assert W_total.shape == (N, N), f"Output shape should be ({N}, {N})"

        # Verify row-normalization
        row_sums = W_total.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(N), atol=1e-5), \
            f"Rows should sum to 1, got min={row_sums.min():.4f}, max={row_sums.max():.4f}"

        # Verify all weights are non-negative
        assert (W_total >= 0).all(), "All weights should be non-negative"

        print(f"[PASS] Combined weights test passed for C={C} ({description})")

    return True


if __name__ == "__main__":
    print("Testing Soft Contrastive Loss module...\n")

    torch.manual_seed(42)

    print("Running soft contrastive loss tests...")
    _test_soft_contrastive_loss()

    print("\nRunning combined weights tests...")
    _test_combined_weights()

    print("\nAll tests completed!")
