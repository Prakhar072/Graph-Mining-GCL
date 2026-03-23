"""
Linear evaluation protocol for the trained encoder.

Evaluates the quality of learned representations by training a linear
classifier on top of frozen encoder outputs.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def linear_evaluation(encoder, X, edge_index, y, train_ratio=0.1, n_runs=10, device='cpu'):
    """
    Linear evaluation protocol.

    For each run:
    - Extract encoder representations H = encoder(X, edge_index)
    - Split nodes into train (train_ratio) and test (1 - train_ratio)
    - Fit LogisticRegression on training representations
    - Evaluate accuracy on test representations
    - Return mean and std accuracy over n_runs

    Args:
        encoder (nn.Module): Trained encoder model
        X (torch.Tensor): Feature matrix
        edge_index (torch.Tensor): Edge indices
        y (torch.Tensor): Node labels
        train_ratio (float): Fraction of nodes for training (default 0.1)
        n_runs (int): Number of evaluation runs (default 10)
        device (str): Device to use

    Returns:
        dict: Contains mean_acc, std_acc, all_accs
    """
    # request knowledge: what is this eval() function doing? why do we need it?
    encoder.eval()
    device = torch.device(device)

    # Extract representations with no gradient
    with torch.no_grad():
        h = encoder(X.to(device), edge_index.to(device))
        h = h.cpu().numpy()
    #request change: more cpu hardcoding and should be converting to numpy here instead of later
    y_np = y.cpu().numpy()
    N = len(y_np)

    accuracies = []

    for run in range(n_runs):
        # Random train/test split
        indices = np.random.permutation(N)
        train_size = int(train_ratio * N)

        train_idx = indices[:train_size]
        test_idx = indices[train_size:]

        X_train = h[train_idx]
        y_train = y_np[train_idx]
        X_test = h[test_idx]
        y_test = y_np[test_idx]

        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=run)
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    return {
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'all_accs': accuracies,
    }


def evaluate_multiple_runs(encoder, X, edge_index, y, n_runs=5, device='cpu'):
    """
    Run multiple evaluation runs with different train ratios.

    Args:
        encoder (nn.Module): Trained encoder
        X (torch.Tensor): Features
        edge_index (torch.Tensor): Edge indices
        y (torch.Tensor): Labels
        n_runs (int): Evaluations per train ratio
        device (str): Device

    Returns:
        dict: Results for each train ratio
    """
    train_ratios = [0.05, 0.1, 0.2]
    results = {}

    for ratio in train_ratios:
        print(f"\nEvaluating with train_ratio={ratio}...")
        eval_result = linear_evaluation(encoder, X, edge_index, y, train_ratio=ratio, n_runs=n_runs, device=device)

        results[ratio] = eval_result
        print(f"  Mean accuracy: {eval_result['mean_acc']:.4f} +/- {eval_result['std_acc']:.4f}")

    return results


def evaluate_with_different_train_sizes(encoder, X, edge_index, y, device='cpu'):
    """
    Evaluate using full training data (semi-supervised setting).

    Uses pre-defined train/val/test splits if available, or creates random splits.

    Args:
        encoder (nn.Module): Trained encoder
        X (torch.Tensor): Features
        edge_index (torch.Tensor): Edge indices
        y (torch.Tensor): Labels
        device (str): Device

    Returns:
        float: Final test accuracy
    """
    encoder.eval()
    device = torch.device(device)

    # Extract representations
    with torch.no_grad():
        h = encoder(X.to(device), edge_index.to(device))
        h = h.cpu().numpy()

    y_np = y.cpu().numpy()

    # Use PyTorch Geometric's standard splits if available
    N = len(y_np)

    # Standard split: 60% train, 20% val, 20% test
    n_train = int(0.6 * N)
    n_val = int(0.2 * N)

    indices = np.arange(N)
    np.random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train = h[train_idx]
    y_train = y_np[train_idx]
    X_val = h[val_idx]
    y_val = y_np[val_idx]
    X_test = h[test_idx]
    y_test = y_np[test_idx]

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Get predictions on validation set to optionally tune hyperparameters
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    # Evaluate on test set
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    return test_acc


if __name__ == "__main__":
    # Test evaluation module
    print("Testing linear evaluation module...\n")

    # Create dummy data
    N, D, C = 100, 50, 5
    X = torch.randn(N, D)
    y = torch.randint(0, C, (N,))

    # Create dummy encoder
    encoder = nn.Sequential(
        nn.Linear(D, 32),
        nn.ReLU(),
        nn.Linear(32, 16)
    )

    # Create dummy edge_index
    edge_index = torch.randint(0, N, (2, 200))

    print("Running linear evaluation...")
    results = linear_evaluation(encoder, X, edge_index, y, train_ratio=0.1, n_runs=3)

    print(f"Mean accuracy: {results['mean_acc']:.4f} +/- {results['std_acc']:.4f}")
    print("Test passed!")
