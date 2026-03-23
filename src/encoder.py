"""
Graph Neural Network encoder for contrastive learning.

Implements a GCN encoder with projection head for contrastive pre-training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    #request knowledge: why this architecture? why 2 layers? why relu and dropout in between?
    """
    Graph Convolutional Network encoder with two layers.

    Architecture:
        - GCN layer 1: in_dim -> hidden_dim (with ReLU)
        - Dropout
        - GCN layer 2: hidden_dim -> out_dim

    Args:
        in_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        out_dim (int): Output representation dimension
        dropout (float): Dropout probability
    """

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(GCNEncoder, self).__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)       
        self.dropout = dropout

    def forward(self, X, edge_index):
        """
        Forward pass through the encoder.

        Args:
            X (torch.Tensor): Node feature matrix, shape (N, in_dim)
            edge_index (torch.Tensor): Edge indices, shape (2, num_edges)

        Returns:
            torch.Tensor: Node representations, shape (N, out_dim)
        """
        # First GCN layer
        h = self.conv1(X, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training) #only during training

        # Second GCN layer
        h = self.conv2(h, edge_index)

        return h

#request knowledge: why do we need a projection head? why not just use the encoder output for contrastive loss?
class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.

    A two-layer MLP that projects encoder representations to a lower-dimensional
    space for computing contrastive loss. Discarded during downstream evaluation.

    Args:
        in_dim (int): Input dimension (encoder output dimension)
        hidden_dim (int): Hidden layer dimension
    """

    def __init__(self, in_dim, hidden_dim):
        super(ProjectionHead, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, h):
        """
        Forward pass through projection head.

        Args:
            h (torch.Tensor): Encoder representations, shape (N, in_dim) or (B, in_dim)

        Returns:
            torch.Tensor: Projected representations, same shape as input
        """
        z = self.fc1(h)
        z = F.relu(z)
        z = self.fc2(z)

        return z


class ContrastiveModel(nn.Module):
    """
    Full model combining encoder and projection head.

    Used during pre-training. The encoder alone is used for downstream evaluation.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, proj_dim, dropout=0.1):
        super(ContrastiveModel, self).__init__()

        self.encoder = GCNEncoder(in_dim, hidden_dim, out_dim, dropout)
        self.projection_head = ProjectionHead(out_dim, proj_dim)

    def forward(self, X, edge_index, return_h=False):
        """
        Forward pass through encoder and projection head.

        Args:
            X (torch.Tensor): Node feature matrix
            edge_index (torch.Tensor): Edge indices
            return_h (bool): If True, return both h (encoder output) and z (projection)

        Returns:
            torch.Tensor: Projected representations z of shape (N, out_dim), or
            tuple: (h, z) if return_h=True
        """
        h = self.encoder(X, edge_index)
        z = self.projection_head(h)

        if return_h:
            return h, z

        return z


if __name__ == "__main__":
    # Test encoder and projection head
    print("Testing encoder components...\n")

    # Create test data
    N = 100
    in_dim = 50
    hidden_dim = 64
    out_dim = 32
    num_edges = 200

    X = torch.randn(N, in_dim)
    edge_index = torch.randint(0, N, (2, num_edges))

    print(f"Input: N={N}, in_dim={in_dim}, num_edges={num_edges}")
    print(f"Encoder dims: {in_dim} -> {hidden_dim} -> {out_dim}")

    # Test GCN encoder
    print("\nTesting GCNEncoder:")
    encoder = GCNEncoder(in_dim, hidden_dim, out_dim, dropout=0.1)
    h = encoder(X, edge_index)
    print(f"  Output shape: {h.shape}")
    assert h.shape == (N, out_dim), f"Expected shape ({N}, {out_dim}), got {h.shape}"
    print("  [PASS]")

    # Test projection head
    print("\nTesting ProjectionHead:")
    proj_dim = 64
    proj_head = ProjectionHead(out_dim, proj_dim)
    z = proj_head(h)
    print(f"  Output shape: {z.shape}")
    assert z.shape == (N, out_dim), f"Expected shape ({N}, {out_dim}), got {z.shape}"
    print("  [PASS]")

    # Test full contrastive model
    print("\nTesting ContrastiveModel:")
    model = ContrastiveModel(in_dim, hidden_dim, out_dim, proj_dim, dropout=0.1)
    z = model(X, edge_index)
    h, z = model(X, edge_index, return_h=True)
    print(f"  Encoder output shape: {h.shape}")
    print(f"  Projection output shape: {z.shape}")
    assert h.shape == (N, out_dim), f"Expected encoder shape ({N}, {out_dim}), got {h.shape}"
    assert z.shape == (N, out_dim), f"Expected projection shape ({N}, {out_dim}), got {z.shape}"
    print("  [PASS]")

    # Test gradient flow
    print("\nTesting gradient flow:")
    X.requires_grad = True
    z = model(X, edge_index)
    loss = z.sum()
    loss.backward()
    assert X.grad is not None, "Gradient should flow to input"
    print("  [PASS] Gradients flow correctly")

    print("\nAll encoder tests passed!")
