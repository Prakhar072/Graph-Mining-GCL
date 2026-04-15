"""
Full training loop for conflict-conditioned graph contrastive learning.

Implements pre-training with soft contrastive loss and fine-tuning with
discriminator-based calibration.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import os

from .config import get_config
from .dataset import load_dataset
from .augment import drop_edges, mask_features
from .encoder import GCNEncoder, ProjectionHead, ContrastiveModel
from .conflict import compute_conflict_index
from .weights import compute_structural_weights, compute_attribute_weights
from .loss import soft_contrastive_loss, compute_combined_weights
from .discriminator import (
    Discriminator,
    get_eigenvectors,
    build_fusion_vectors,
    get_pair_scores,
    build_pretrain_pairs,
    build_finetune_pairs,
)
from .loss import balanced_softmax_loss


def set_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_checkpoints_dir(cfg):
    """Create checkpoint directory, organized by dataset."""
    checkpoint_root = Path(cfg.checkpoint_dir)
    checkpoint_dir = checkpoint_root / cfg.dataset
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(encoder, epoch, checkpoint_dir, name='encoder'):
    """Save model checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / f'{name}_epoch_{epoch}.pt'
    torch.save(encoder.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """Load model from checkpoint with validation."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Validate checkpoint compatibility by checking conv1.lin.weight shape
    try:
        # Get expected weight shape from model
        expected_weight_shape = model.conv1.lin.weight.shape
        
        # Get weight shape from checkpoint
        checkpoint_weight_shape = state_dict['conv1.lin.weight'].shape
        
        if expected_weight_shape != checkpoint_weight_shape:
            raise RuntimeError(
                f"Checkpoint incompatible: expected conv1.lin.weight shape {expected_weight_shape}, "
                f"but checkpoint has {checkpoint_weight_shape}"
            )
    except Exception as e:
        raise RuntimeError(f"Checkpoint validation failed: {e}")
    
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def find_latest_checkpoint(checkpoint_dir, dataset_name):
    """Find the latest checkpoint (preferring fine-tuned over pre-trained) for a dataset."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    
    # Create dataset-specific subdirectory name
    dataset_checkpoint_dir = checkpoint_path / dataset_name
    
    # If dataset-specific checkpoints exist, use those
    if dataset_checkpoint_dir.exists():
        checkpoint_path = dataset_checkpoint_dir
    
    # First priority: Look for fine-tuning checkpoints (encoder_finetune_iter_*.pt)
    finetune_checkpoints = list(checkpoint_path.glob('encoder_finetune_iter_*.pt'))
    if finetune_checkpoints:
        # Return the most recent one
        latest = sorted(finetune_checkpoints, key=lambda p: p.stat().st_mtime)[-1]
        print(f"Using fine-tuned checkpoint: {latest.name}")
        return latest
    
    # Second priority: Look for the final pretrain checkpoint
    final_checkpoints = list(checkpoint_path.glob('encoder_pretrain_final*.pt'))
    if final_checkpoints:
        # Return the most recent one
        latest = sorted(final_checkpoints, key=lambda p: p.stat().st_mtime)[-1]
        print(f"Using pre-trained checkpoint: {latest.name}")
        return latest
    
    # Fall back to any encoder checkpoint
    all_checkpoints = list(checkpoint_path.glob('encoder*.pt'))
    if all_checkpoints:
        return sorted(all_checkpoints, key=lambda p: p.stat().st_mtime)[-1]
    
    return None


def load_pretrained_encoder(dataset_name, device='cpu', checkpoint_dir='./checkpoints', **kwargs):
    """Load pre-trained encoder from checkpoint."""
    from .encoder import GCNEncoder
    from .config import get_config
    from .dataset import load_dataset
    
    # Get config
    cfg = get_config(dataset_name, **kwargs)
    
    # Load dataset to get dimensions
    A_norm, X, y, edge_index, N, D = load_dataset(dataset_name)
    cfg.in_dim = D
    
    # Create encoder
    encoder = GCNEncoder(cfg.in_dim, cfg.hidden_dim, cfg.out_dim, cfg.dropout).to(device)
    
    # Find checkpoints in priority order
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Look for dataset-specific subdirectory first
    dataset_checkpoint_dir = checkpoint_path / dataset_name
    search_path = dataset_checkpoint_dir if dataset_checkpoint_dir.exists() else checkpoint_path
    
    # Try fine-tuning checkpoints first (most recent to oldest)
    finetune_checkpoints = sorted(
        search_path.glob('encoder_finetune_iter_*.pt'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    # Try pretrain final checkpoints next
    pretrain_final_checkpoints = sorted(
        search_path.glob('encoder_pretrain_final*.pt'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    all_candidates = finetune_checkpoints + pretrain_final_checkpoints
    
    for checkpoint_candidate in all_candidates:
        try:
            print(f"Trying checkpoint: {checkpoint_candidate.name}...", end=" ")
            encoder = load_checkpoint(encoder, str(checkpoint_candidate), device=device)
            print("✓")
            encoder.eval()
            return encoder, cfg
        except RuntimeError as e:
            print(f"✗ (Skipped: {str(e)[:60]}...)")
            # Recreate fresh encoder for next attempt
            encoder = GCNEncoder(cfg.in_dim, cfg.hidden_dim, cfg.out_dim, cfg.dropout).to(device)
            continue
    
    # No compatible checkpoint found
    raise FileNotFoundError(
        f"No compatible checkpoint found in {search_path}. "
        f"Please train the model first without --evaluate"
    )

def find_resume_stage(checkpoint_dir, dataset_name):
    """Determine resume stage from existing checkpoints.
    
    Returns: (stage, checkpoint_info)
        stage: 'finetune', 'pretrain_final', 'pretrain', or 'start'
        checkpoint_info: 
            - For 'finetune': (encoder_path, discriminator_path, iteration)
            - For 'pretrain_final': encoder_path
            - For 'pretrain': epoch number
            - For 'start': None
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return 'start', None
    
    # Look in dataset-specific subdirectory
    dataset_checkpoint_dir = checkpoint_path / dataset_name
    search_path = dataset_checkpoint_dir if dataset_checkpoint_dir.exists() else checkpoint_path
    
    # Check for fine-tuning checkpoints
    finetune_checkpoints = sorted(
        search_path.glob('encoder_finetune_iter_*.pt'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if finetune_checkpoints:
        # Extract iteration number and find corresponding discriminator checkpoint
        latest = finetune_checkpoints[0]
        import re
        match = re.search(r'encoder_finetune_iter_(\d+)', latest.name)
        if match:
            iteration = int(match.group(1))
            # Look for corresponding discriminator checkpoint
            disc_checkpoint = search_path / f'discriminator_finetune_iter_{iteration}_epoch_{iteration}.pt'
            if not disc_checkpoint.exists():
                # Maybe it doesn't exist yet, that's okay
                disc_checkpoint = None
            return 'finetune', (str(latest), str(disc_checkpoint) if disc_checkpoint else None, iteration)
    
    # Check for pretrain final checkpoint
    pretrain_final_checkpoints = sorted(
        search_path.glob('encoder_pretrain_final*.pt'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if pretrain_final_checkpoints:
        return 'pretrain_final', str(pretrain_final_checkpoints[0])
    
    # Check for pretrain checkpoints (encoder_pretrain_epoch_X.pt)
    pretrain_checkpoints = sorted(
        search_path.glob('encoder_pretrain_epoch_*.pt'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if pretrain_checkpoints:
        # Extract epoch number from filename
        latest = pretrain_checkpoints[0]
        import re
        match = re.search(r'encoder_pretrain_epoch_(\d+)', latest.name)
        if match:
            epoch = int(match.group(1))
            return 'pretrain', epoch
    
    return 'start', None


def pretrain_encoder(model, cfg, X, edge_index, W_total, device, checkpoint_dir, resume_from_epoch=None):
    """Pre-train encoder with soft contrastive loss, optionally resuming from checkpoint.
    
    Args:
        resume_from_epoch: If provided, resume from this epoch (skip earlier epochs)
    """
    print("\n" + "="*60)
    print("PRE-TRAINING ENCODER")
    if resume_from_epoch:
        print(f"(Resuming from epoch {resume_from_epoch})")
    print("="*60)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr_enc)
    
    start_epoch = resume_from_epoch if resume_from_epoch else 0
    
    # Load checkpoint if resuming
    if resume_from_epoch and checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / f'encoder_pretrain_epoch_{resume_from_epoch}.pt'
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path.name}")
            model.encoder = load_checkpoint(model.encoder, str(checkpoint_path), device=device)
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}, starting fresh from epoch {resume_from_epoch}")

    N = X.shape[0]
    batch_size = cfg.batch_size

    for epoch in range(start_epoch, cfg.pretrain_enc_epochs):
        # Re-shuffle node order each epoch for diverse negative sampling
        batch_indices = torch.randperm(N).split(batch_size)
        total_loss = 0

        for batch_idx, batch_nodes in enumerate(batch_indices):
            batch_nodes = batch_nodes.to(device)
            B = len(batch_nodes)

            # Create augmented views
            edge_idx_u = drop_edges(edge_index, N, cfg.p_e).to(device)
            X_u = mask_features(X, cfg.p_f).to(device)
            edge_idx_v = drop_edges(edge_index, N, cfg.p_e).to(device)
            X_v = mask_features(X, cfg.p_f).to(device)

            # Get representations for both views
            z_u = model(X_u, edge_idx_u)
            z_v = model(X_v, edge_idx_v)
            z_u_batch = z_u[batch_nodes]
            z_v_batch = z_v[batch_nodes]

            # Get soft weights for batch
            W_batch_sliced = W_total[batch_nodes][:, batch_nodes]
            row_sums = W_batch_sliced.sum(dim=1, keepdim=True).clamp(min=1e-8)
            W_batch_sliced = W_batch_sliced / row_sums

            # Compute cross-view contrastive loss
            loss = soft_contrastive_loss(
                z_u_batch,
                z_v_batch,
                W_batch_sliced.to(device),
                cfg.tau,
                cfg.k,
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(batch_indices)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{cfg.pretrain_enc_epochs}: Loss = {avg_loss:.6f}")

        if cfg.save_checkpoints and (epoch + 1) % 50 == 0:
            save_checkpoint(model.encoder, epoch + 1, checkpoint_dir, 'encoder_pretrain')

    print("Encoder pre-training completed!")

    if cfg.save_checkpoints:
        save_checkpoint(model.encoder, cfg.pretrain_enc_epochs, checkpoint_dir, 'encoder_pretrain_final')


def pretrain_discriminator(discriminator, optimizer, encoder, cfg, X, edge_index, A, C_struct, device):
    """Pre-train discriminator with balanced softmax loss."""
    print("\n" + "="*60)
    print("PRE-TRAINING DISCRIMINATOR")
    print("="*60)

    N = X.shape[0]

    for epoch in range(cfg.pretrain_disc_epochs):
        # Get current encoder output (frozen)
        with torch.no_grad():
            H = encoder(X.to(device), edge_index.to(device)).detach()

        # Build fusion vectors
        C_struct_dev = C_struct.to(device)
        Z = build_fusion_vectors(H, C_struct_dev)

        # Build pairs
        pos_pairs, neg_pairs = build_pretrain_pairs(A, X, k=cfg.k_nn)
        pos_pairs = pos_pairs.to(device)
        neg_pairs = neg_pairs.to(device)

        if pos_pairs.shape[0] == 0 or neg_pairs.shape[0] == 0:
            print(f"Epoch {epoch + 1}: Skipping (insufficient pairs)")
            continue

        # Get scores
        scores_pos = get_pair_scores(discriminator, Z, pos_pairs)
        scores_neg = get_pair_scores(discriminator, Z, neg_pairs)

        # Combine scores and labels
        scores = torch.cat([scores_pos, scores_neg])
        labels = torch.cat([
            torch.ones(scores_pos.shape[0], device=device),
            torch.zeros(scores_neg.shape[0], device=device)
        ])

        # Compute loss
        loss = balanced_softmax_loss(scores, labels, pos_pairs.shape[0], neg_pairs.shape[0])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{cfg.pretrain_disc_epochs}: Loss = {loss.item():.6f}")

    print("Discriminator pre-training completed!")


def finetune_phase(
    encoder,
    discriminator,
    optimizer_enc,
    optimizer_disc,
    cfg,
    X,
    edge_index,
    A,
    W_total,
    W_s,
    C,
    C_struct,
    device,
    checkpoint_dir,
    resume_from_iteration=None,
):
    """Fine-tuning phase with discriminator-based weight calibration.
    
    Args:
        resume_from_iteration: If provided, resume from this iteration (skip earlier iterations)
    """
    print("\n" + "="*60)
    print("FINE-TUNING PHASE")
    if resume_from_iteration:
        print(f"(Resuming from iteration {resume_from_iteration})")
    print("="*60)

    N = X.shape[0]
    batch_size = cfg.batch_size
    
    start_iteration = resume_from_iteration if resume_from_iteration else 0

    for iteration in range(start_iteration, cfg.n_iterations):
        print(f"\nIteration {iteration + 1}/{cfg.n_iterations}")

        # Refresh attribute supervision periodically using current encoder embeddings.
        # This makes W_a (and therefore W_total) dynamic rather than fixed to raw features.
        if iteration % 5 == 0:
            print("  Refreshing dynamic attribute weights (W_a) ...")
            encoder.eval()
            with torch.no_grad():
                H_current = encoder(X.to(device), edge_index.to(device))
                H_norm = F.normalize(H_current, dim=1)
                W_a_updated = torch.mm(H_norm, H_norm.t())
                W_a_updated = torch.relu(W_a_updated)

                row_sums = W_a_updated.sum(dim=1, keepdim=True)
                zero_rows = row_sums.squeeze(1) <= 1e-12
                if zero_rows.any():
                    idx = torch.nonzero(zero_rows, as_tuple=False).squeeze(1)
                    W_a_updated[idx, idx] = 1.0
                    row_sums = W_a_updated.sum(dim=1, keepdim=True)

                W_a_updated = W_a_updated / row_sums.clamp(min=1e-8)

                W_total = compute_combined_weights(
                    W_s.to(device),
                    W_a_updated,
                    C,
                    alpha=cfg.alpha,
                    beta=cfg.beta,
                ).to(device)

        # STEP A: Update encoder
        print("  Updating encoder...")

        encoder.eval()  # Sample noise from frozen encoder during augmentation
        discriminator.eval()

        for enc_epoch in range(cfg.finetune_enc_epochs):
            encoder.train()

            batch_indices = torch.randperm(N).split(batch_size)
            epoch_loss = 0
            num_batches = 0

            for batch_nodes in batch_indices:
                batch_nodes = batch_nodes.to(device)
                B = len(batch_nodes)

                # Create augmented views
                edge_idx_u = drop_edges(edge_index, N, cfg.p_e).to(device)
                X_u = mask_features(X, cfg.p_f).to(device)
                edge_idx_v = drop_edges(edge_index, N, cfg.p_e).to(device)
                X_v = mask_features(X, cfg.p_f).to(device)

                # Get representations (with gradients enabled for encoder update)
                H_u = encoder(X_u, edge_idx_u)
                H_v = encoder(X_v, edge_idx_v)

                C_struct_dev = C_struct.to(device)
                with torch.no_grad():
                    discriminator.eval()
                    C_struct_batch = C_struct_dev[batch_nodes]
                    Z_u = build_fusion_vectors(H_u[batch_nodes].detach(), C_struct_batch)
                    pair_indices = torch.stack(
                        [
                            torch.arange(B, device=device).repeat_interleave(B),
                            torch.arange(B, device=device).repeat(B),
                        ],
                        dim=1,
                    )
                    calibration_scores = get_pair_scores(discriminator, Z_u, pair_indices)

                # Build calibrated weights using SOFT thresholding (not hard)
                # Use sigmoid to convert scores to soft weights in [0, 1]
                # This prevents complete zeroing out of weight matrix
                soft_weights = torch.sigmoid((calibration_scores - cfg.eta) * 10)  # Sharper sigmoid centered at eta
                W_batch_sliced = W_total[batch_nodes][:, batch_nodes].to(device)
                row_sums = W_batch_sliced.sum(dim=1, keepdim=True).clamp(min=1e-8)
                W_batch_sliced = W_batch_sliced / row_sums
                W_cali = W_batch_sliced * soft_weights.reshape(B, B)

                # Get batch representations for both views
                H_u_batch = H_u[batch_nodes]
                H_v_batch = H_v[batch_nodes]

                # Compute calibrated cross-view loss
                loss = soft_contrastive_loss(H_u_batch, H_v_batch, W_cali, cfg.tau, cfg.k)

                # Backward pass
                optimizer_enc.zero_grad()
                loss.backward()
                optimizer_enc.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            if (enc_epoch + 1) % 5 == 0:
                print(f"    Encoder epoch {enc_epoch + 1}/{cfg.finetune_enc_epochs}: Loss = {avg_loss:.6f}")

        # STEP B: Update discriminator
        print("  Updating discriminator...")

        # Get current H
        with torch.no_grad():
            H_current = encoder(X.to(device), edge_index.to(device)).detach()

        # Build finetune pairs
        pos_pairs, neg_pairs = build_finetune_pairs(A, H_current, k=cfg.k_nn)
        pos_pairs = pos_pairs.to(device)
        neg_pairs = neg_pairs.to(device)

        if pos_pairs.shape[0] == 0 or neg_pairs.shape[0] == 0:
            print(f"    Iteration {iteration + 1}: Skipping disc update (insufficient pairs)")
            continue

        # Train discriminator
        for disc_epoch in range(cfg.finetune_disc_epochs):
            discriminator.train()

            # Build fusion vectors
            C_struct_dev = C_struct.to(device)
            Z = build_fusion_vectors(H_current, C_struct_dev)

            # Get scores
            scores_pos = get_pair_scores(discriminator, Z, pos_pairs)
            scores_neg = get_pair_scores(discriminator, Z, neg_pairs)

            # Combine
            scores = torch.cat([scores_pos, scores_neg])
            labels = torch.cat([
                torch.ones(scores_pos.shape[0], device=device),
                torch.zeros(scores_neg.shape[0], device=device)
            ])

            # Loss and update
            loss = balanced_softmax_loss(scores, labels, pos_pairs.shape[0], neg_pairs.shape[0])

            optimizer_disc.zero_grad()
            loss.backward()
            optimizer_disc.step()

            if (disc_epoch + 1) % 3 == 0:
                print(f"    Discriminator epoch {disc_epoch + 1}/{cfg.finetune_disc_epochs}: Loss = {loss.item():.6f}")

        if cfg.save_checkpoints:
            save_checkpoint(encoder, iteration + 1, checkpoint_dir, f'encoder_finetune_iter_{iteration + 1}')
            save_checkpoint(discriminator, iteration + 1, checkpoint_dir, f'discriminator_finetune_iter_{iteration + 1}')


def train(dataset_name, device='cpu', resume=False, **kwargs):
    """
    Full training pipeline.

    Args:
        dataset_name (str): Dataset name
        device (str): Device to use ('cpu' or 'cuda')
        resume (bool): Resume from latest checkpoint if available
        **kwargs: Override config parameters
    """
    # Setup
    cfg = get_config(dataset_name, **kwargs)
    cfg.device = device
    set_seed(cfg.seed)

    device = torch.device(device)
    checkpoint_dir = create_checkpoints_dir(cfg) if cfg.save_checkpoints else None

    print("="*60)
    print(f"Training on {dataset_name} dataset")
    print(f"Device: {device}")
    print("="*60)

    # Load data
    print("\nLoading dataset...")
    A_norm, X, y, edge_index, N, D = load_dataset(dataset_name)

    cfg.in_dim = D
    X = X.to(device)
    edge_index = edge_index.to(device)

    # Use cached dense normalized adjacency from dataset loader
    A_norm_dense = A_norm.to(device)

    # Determine resume stage if requested
    resume_stage = 'start'
    resume_info = None
    if resume and checkpoint_dir:
        resume_stage, resume_info = find_resume_stage(checkpoint_dir, dataset_name)
        if resume_stage != 'start':
            print(f"\n✓ Found existing checkpoints, resuming from stage: {resume_stage}")

    # Compute conflict index
    print("\nComputing conflict index...")
    C = compute_conflict_index(A_norm_dense, X, n_samples=10000)
    print("Conflict index C :", C) 

    # Compute weight matrices
    print("\nComputing structural weights...")
    W_s = compute_structural_weights(A_norm_dense, N, method='ppr', alpha=0.15)

    print("Computing attribute weights...")
    W_a = compute_attribute_weights(X)

    # Combine weights
    print("Combining weights...")
    W_total = compute_combined_weights(W_s, W_a, C, alpha=cfg.alpha, beta=cfg.beta)
    W_total = W_total.to(device)

    # Get eigenvectors (needed for both pre-training and fine-tuning)
    print("Computing structural eigenvectors...")
    C_struct = get_eigenvectors(A_norm_dense.to(device), t=cfg.t, device=device)

    # Create models
    print("\nCreating models...")
    model = ContrastiveModel(cfg.in_dim, cfg.hidden_dim, cfg.out_dim, cfg.proj_dim, cfg.dropout).to(device)
    encoder = model.encoder
    discriminator = Discriminator(cfg.out_dim + cfg.t, cfg.hidden_dim).to(device)
    
    # Check if we need to resume from a checkpoint
    if resume_stage in ['finetune', 'pretrain_final', 'pretrain']:
        if resume_stage == 'finetune':
            encoder_path, disc_path, resume_iteration = resume_info
            print(f"\nLoading fine-tuning checkpoints:")
            print(f"  Encoder: {Path(encoder_path).name}")
            encoder = load_checkpoint(encoder, encoder_path, device=device)
            if disc_path and Path(disc_path).exists():
                print(f"  Discriminator: {Path(disc_path).name}")
                discriminator = load_checkpoint(discriminator, disc_path, device=device)
            else:
                print(f"  Discriminator: Not found, creating fresh")
        elif resume_stage == 'pretrain_final':
            checkpoint_path = resume_info
            print(f"\nLoading pre-trained checkpoint: {Path(checkpoint_path).name}")
            encoder = load_checkpoint(encoder, checkpoint_path, device=device)
        elif resume_stage == 'pretrain':
            resume_epoch = resume_info
            # For pretrain resume, we don't load the checkpoint here
            # It will be loaded in pretrain_encoder function
            print(f"\nResuming pre-training from epoch {resume_epoch}")

    # PRE-TRAINING PHASE
    if resume_stage not in ['pretrain_final', 'finetune']:
        # Need to do pre-training
        resume_epoch = resume_info if resume_stage == 'pretrain' else None
        pretrain_encoder(model, cfg, X, edge_index, W_total, device, checkpoint_dir, resume_from_epoch=resume_epoch)
    else:
        print("\n✓ Skipping pre-training (resuming from later stage)")

    encoder = model.encoder

    # DISCRIMINATOR PRE-TRAINING
    if resume_stage != 'finetune':
        # Only skip discriminator pre-training if resuming from fine-tuning
        optimizer_disc_pretrain = optim.Adam(discriminator.parameters(), lr=cfg.lr_disc)

        pretrain_discriminator(
            discriminator,
            optimizer_disc_pretrain,
            encoder,
            cfg,
            X,
            edge_index,
            A_norm_dense,
            C_struct,
            device,
        )
    else:
        # Skip discriminator pre-training (already done in previous run)
        print("✓ Skipping discriminator pre-training (resuming from fine-tuning)")

    # FINE-TUNING PHASE
    optimizer_enc_finetune = optim.Adam(encoder.parameters(), lr=cfg.lr_enc)
    optimizer_disc_finetune = optim.Adam(discriminator.parameters(), lr=cfg.lr_disc)

    resume_iteration = None
    if resume_stage == 'finetune':
        _, resume_iteration = resume_info
    
    finetune_phase(
        encoder,
        discriminator,
        optimizer_enc_finetune,
        optimizer_disc_finetune,
        cfg,
        X,
        edge_index,
        A_norm_dense,
        W_total,
        W_s,
        C,
        C_struct,
        device,
        checkpoint_dir,
        resume_from_iteration=resume_iteration,
    )

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)

    return encoder, cfg


if __name__ == "__main__":
    # Quick test
    print("Training framework test...")
    # train('cora', device='cpu')
