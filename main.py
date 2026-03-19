"""
Main entry point for conflict-conditioned graph contrastive learning framework.

This script demonstrates how to use the complete framework for:
1. Pre-training with soft contrastive loss
2. Fine-tuning with discriminator-based calibration
3. Linear evaluation on downstream tasks
"""

import sys
import argparse
import torch

#Add src to path
from pathlib import Path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from train import train
from evaluate import linear_evaluation, evaluate_multiple_runs
from config import get_config, print_config
from dataset import load_dataset


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Conflict-Conditioned Graph Contrastive Learning"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        choices=["cora", "citeseer", "pubmed", "chameleon", "squirrel", "actor"],
        help="Dataset name"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=None,
        help="Number of pre-training epochs (overrides config)"
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Temperature parameter (overrides config)"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run linear evaluation after training"
    )

    parser.add_argument(
        "--n-eval-runs",
        type=int,
        default=10,
        help="Number of evaluation runs"
    )

    args = parser.parse_args()

    # Prepare overrides
    kwargs = {"seed": args.seed}
    if args.pretrain_epochs is not None:
        kwargs["pretrain_enc_epochs"] = args.pretrain_epochs
    if args.tau is not None:
        kwargs["tau"] = args.tau

    print("\n" + "="*70)
    print("CONFLICT-CONDITIONED GRAPH CONTRASTIVE LEARNING")
    print("="*70)

    # Get config and print
    cfg = get_config(args.dataset, **kwargs)
    print_config(cfg)

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\nWARNING: CUDA requested but not available. Using CPU.")
        args.device = "cpu"

    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    encoder, cfg = train(args.dataset, device=args.device, **kwargs)

    # Evaluate (optional)
    if args.evaluate:
        print("\n" + "="*70)
        print("LINEAR EVALUATION")
        print("="*70)

        # Load data
        A_norm, X, y, edge_index, N, D = load_dataset(args.dataset)

        # Move to appropriate device
        device = torch.device(args.device)
        X = X.to(device)
        edge_index = edge_index.to(device)

        # Run evaluation
        results = evaluate_multiple_runs(
            encoder,
            X,
            edge_index,
            y,
            n_runs=args.n_eval_runs,
            device=args.device
        )

        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        for ratio, result in results.items():
            print(f"\nTrain ratio {ratio}:")
            print(f"  Mean accuracy: {result['mean_acc']:.4f}")
            print(f"  Std dev:      {result['std_acc']:.4f}")
            print(f"  Raw scores:   {[f'{acc:.4f}' for acc in result['all_accs']]}")

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
