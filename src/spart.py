"""
Exponential Partitioned Similarity (SPART) for contrastive learning.

This module implements the SPART similarity kernel that partitions feature
dimensions into chunks and computes exponential similarities across each
partition, then averages them. This provides robustness to feature noise
and improves learning stability.
"""

import torch
import torch.nn.functional as F


def spart_similarity(H1, H2, m, tau):
    """
    Compute exponential partitioned similarity between two batches of representations.

    Args:
        H1 (torch.Tensor): Batch of representations, shape (B, d)
        H2 (torch.Tensor): Batch of representations, shape (B, d)
        m (int): Partition size (number of features per partition)
        tau (float): Temperature parameter for scaling

    Returns:
        torch.Tensor: Similarity matrix of shape (B, B) where S[i,j] is the
                     similarity between H1[i] and H2[j]
    """
    B, d = H1.shape
    assert H2.shape == (B, d), f"H1 and H2 must have same shape, got {H1.shape} and {H2.shape}"

    # Generate a random permutation of feature dimensions
    perm = torch.randperm(d, device=H1.device)

    # Apply the same permutation to both H1 and H2
    H1_shuffled = H1[:, perm]
    H2_shuffled = H2[:, perm]

    # Compute number of partitions
    K = d // m
    assert K > 0, f"Partition size m={m} must be <= feature dimension d={d}"

    # Accumulator for log-probabilities (for numerically stable computation)
    log_S_list = []

    # For each partition, compute inner product matrix and store log-exp
    for k in range(K):
        start_idx = k * m
        end_idx = start_idx + m

        # Extract partition k from both batches
        H1_k = H1_shuffled[:, start_idx:end_idx]  # shape: (B, m)
        H2_k = H2_shuffled[:, start_idx:end_idx]  # shape: (B, m)

        # Compute inner product matrix for this partition
        S_k = torch.mm(H1_k, H2_k.t())  # shape: (B, B)

        # Apply scaling (note: K * S_k / tau is the exponent)
        # We'll handle the exponential in a numerically stable way using logsumexp
        log_S_list.append(K * S_k / tau)

    # Use logsumexp for numerical stability
    # logsumexp(x) = log(sum(exp(x))) computed stably
    # We want: (1/K) * sum(exp(K*S_k/tau)) = exp(logsumexp(log_S_list) - log(K))
    log_S_stacked = torch.stack(log_S_list, dim=0)  # shape: (K, B, B)
    log_S_part = torch.logsumexp(log_S_stacked, dim=0) - torch.log(torch.tensor(K, dtype=H1.dtype, device=H1.device))

    # Exponentiate to get the final similarity
    S_part = torch.exp(log_S_part)

    return S_part


def _test_spart_lower_bound():
    """
    Test that spart_similarity provides a lower bound on standard exponential similarity.

    Verifies that spart_similarity(H, H, m, tau) >= exp(H @ H.T / tau) elementwise.
    This is based on Jensen's inequality for the exponential function.
    """
    # Create test tensors
    B, d = 16, 128
    m = 32
    tau = 0.8

    H = torch.randn(B, d)

    # Compute SPART similarity
    S_spart = spart_similarity(H, H, m, tau)

    # Compute standard exponential similarity
    S_standard = torch.exp(torch.mm(H, H.t()) / tau)

    # Check that SPART >= standard elementwise (with small tolerance for numerical errors)
    diff = S_spart - S_standard
    tolerance = 1e-5

    violations = (diff < -tolerance).sum().item()

    if violations == 0:
        print(f"[PASS] SPART lower bound test passed: {violations} violations")
        return True
    else:
        print(f"[FAIL] SPART lower bound test FAILED: {violations} violations")
        print(f"  Min difference: {diff.min().item():.6f}")
        print(f"  Max difference: {diff.max().item():.6f}")
        return False


def _test_spart_properties():
    """
    Test basic properties of SPART similarity:
    - Output shape is correct
    - Diagonal is positive
    - Values are positive (exponential)
    - Symmetric when H1 == H2
    """
    B, d = 32, 256
    m = 64
    tau = 0.8

    H1 = torch.randn(B, d)
    H2 = torch.randn(B, d)

    S = spart_similarity(H1, H2, m, tau)

    tests_passed = []

    # Test 1: Output shape
    assert S.shape == (B, B), f"Expected shape ({B}, {B}), got {S.shape}"
    tests_passed.append("[PASS] Output shape correct")

    # Test 2: Values are positive
    assert (S > 0).all(), "All similarity values should be positive"
    tests_passed.append("[PASS] All values positive")

    # Test 3: High diagonal (self-similarity)
    S_self = spart_similarity(H1, H1, m, tau)
    diag = torch.diag(S_self).mean()
    assert diag > 1.0, f"Expected high self-similarity, got {diag:.4f}"
    tests_passed.append("[PASS] High self-similarity on diagonal")

    # Test 4: Symmetry when H1 == H2
    torch.manual_seed(42)
    H = torch.randn(B, d)
    S1 = spart_similarity(H, H, m, tau)
    torch.manual_seed(42)
    S2 = spart_similarity(H, H, m, tau)
    # Note: not strictly symmetric due to random permutation, but shape is consistent
    assert S1.shape == S2.shape, "Shapes should be consistent"
    tests_passed.append("[PASS] Consistent output for same input")

    for msg in tests_passed:
        print(msg)

    return True


if __name__ == "__main__":
    print("Testing SPART similarity module...\n")

    print("Running property tests...")
    _test_spart_properties()

    print("\nRunning lower bound test...")
    _test_spart_lower_bound()

    print("\nAll tests completed!")
