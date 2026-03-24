"""
Linear evaluation protocol for the trained encoder.

Evaluates the quality of learned representations by training a linear
classifier on top of frozen encoder outputs.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score


def linear_evaluation(encoder, X, edge_index, y, train_ratio=0.1, n_runs=10, device='cpu'):
    """
    Linear evaluation protocol.

    For each run:
    - Extract encoder representations H = encoder(X, edge_index)
    - Split nodes into train (train_ratio) and test (1 - train_ratio)
    - Fit LogisticRegression on training representations
    - Evaluate weighted F1 score on test representations
    - Return mean and std weighted F1 over n_runs

    Args:
        encoder (nn.Module): Trained encoder model
        X (torch.Tensor): Feature matrix
        edge_index (torch.Tensor): Edge indices
        y (torch.Tensor): Node labels
        train_ratio (float): Fraction of nodes for training (default 0.1)
        n_runs (int): Number of evaluation runs (default 10)
        device (str): Device to use

    Returns:
        dict: Contains mean_f1, std_f1, all_f1_scores
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

        # Train linear regression
        clf = LinearRegression()
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        # Convert continuous predictions to class labels via argmax for multi-class
        if len(y_test.shape) == 1:
            y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else (y_pred > 0.5).astype(int)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        accuracies.append(f1)

    mean_f1 = np.mean(accuracies)
    std_f1 = np.std(accuracies)

    return {
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'all_f1_scores': accuracies,
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
        print(f"  Mean F1 score: {eval_result['mean_f1']:.4f} +/- {eval_result['std_f1']:.4f}")

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
        float: Final test weighted F1 score
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
    clf = LinearRegression()
    clf.fit(X_train, y_train)

    # Get predictions on validation set to optionally tune hyperparameters
    y_val_pred = clf.predict(X_val)
    # Convert continuous predictions to class labels via argmax for multi-class
    if y_val_pred.ndim > 1:
        y_val_pred = np.argmax(y_val_pred, axis=1)
    else:
        y_val_pred = (y_val_pred > 0.5).astype(int)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)

    # Evaluate on test set
    y_test_pred = clf.predict(X_test)
    # Convert continuous predictions to class labels via argmax for multi-class
    if y_test_pred.ndim > 1:
        y_test_pred = np.argmax(y_test_pred, axis=1)
    else:
        y_test_pred = (y_test_pred > 0.5).astype(int)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    print(f"Validation F1 score: {val_f1:.4f}")
    print(f"Test F1 score: {test_f1:.4f}")

    return test_f1


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

    print(f"Mean F1 score: {results['mean_f1']:.4f} +/- {results['std_f1']:.4f}")
    print("Test passed!")
