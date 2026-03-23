"""
Configuration management for graph contrastive learning framework.

Provides dataset-specific hyperparameters and a unified configuration interface.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Config:
    """Configuration dataclass for all hyperparameters."""

    # Dataset and training
    dataset: str = 'cora'
    device: str = 'cpu'
    seed: int = 42

    # Data dimensions
    in_dim: int = None  # Set by dataset
    hidden_dim: int = 64
    out_dim: int = 256
    proj_dim: int = 256

    # Augmentation
    p_e: float = 0.3  # Edge drop probability
    p_f: float = 0.3  # Feature mask probability

    # Contrastive learning
    tau: float = 0.8  # Temperature
    m: int = 128  # SPART partition size

    # Discriminator
    t: int = 64  # Number of eigenvectors

    # Conflict-based weighting
    alpha: float = 5.0  # Sigmoid sharpness
    beta: float = 0.5  # Sigmoid midpoint
    eta: float = 0.6  # Discriminator threshold

    # Training phases
    pretrain_enc_epochs: int = 300
    pretrain_disc_epochs: int = 25
    n_iterations: int = 20
    finetune_enc_epochs: int = 15
    finetune_disc_epochs: int = 10

    # Optimizers
    lr_enc: float = 1e-3
    lr_disc: float = 8e-3
    dropout: float = 0.1

    # Batch processing
    batch_size: int = 1024
    k_nn: int = 10

    # Checkpoint-saving options
    save_checkpoints: bool = True
    checkpoint_dir: str = './checkpoints'


DATASET_CONFIGS = {
    #dictionary of dictionaries, used to override defaults after the dataset is loaded.
    'cora': {
        'tau': 0.8,
        'm': 128,
        'pretrain_enc_epochs': 300,
        'n_iterations': 20,
        'finetune_enc_epochs': 15,
        'finetune_disc_epochs': 10,
        'alpha': 5.0,
        'beta': 0.5,
        'lr_enc': 1e-3,
        'lr_disc': 8e-3,
    },
    'citeseer': {
        'tau': 0.8,
        'm': 128,
        'pretrain_enc_epochs': 300,
        'n_iterations': 20,
        'finetune_enc_epochs': 15,
        'finetune_disc_epochs': 10,
        'alpha': 5.0,
        'beta': 0.5,
        'lr_enc': 1e-3,
        'lr_disc': 8e-3,
    },
    'pubmed': {
        'tau': 0.8,
        'm': 128,
        'pretrain_enc_epochs': 300,
        'n_iterations': 20,
        'finetune_enc_epochs': 15,
        'finetune_disc_epochs': 10,
        'alpha': 5.0,
        'beta': 0.5,
        'lr_enc': 1e-3,
        'lr_disc': 8e-3,
    },
    'chameleon': {
        'tau': 1.2,  # Higher temp for heterophilic
        'm': 128,
        'pretrain_enc_epochs': 500,  # Increased for heterophilic
        'n_iterations': 35,  # Increased iterations
        'finetune_enc_epochs': 20,  # Longer fine-tuning per iteration
        'finetune_disc_epochs': 15,  # Longer discriminator tuning
        'alpha': 5.0,
        'beta': 0.5,
        'lr_enc': 1.2e-3,  # Slightly higher LR
        'lr_disc': 1e-2,  # Slightly higher LR
    },
    'squirrel': {
        'tau': 1.2,  # Higher temp for heterophilic
        'm': 128,
        'pretrain_enc_epochs': 500,  # Increased for heterophilic
        'n_iterations': 35,  # Increased iterations
        'finetune_enc_epochs': 20,  # Longer fine-tuning per iteration
        'finetune_disc_epochs': 15,  # Longer discriminator tuning
        'alpha': 5.0,
        'beta': 0.5,
        'lr_enc': 1.2e-3,  # Slightly higher LR
        'lr_disc': 1e-2,  # Slightly higher LR
    },
    'actor': {
        'tau': 1.2,  # Higher temp for heterophilic
        'm': 128,
        'pretrain_enc_epochs': 500,  # Increased for heterophilic
        'n_iterations': 40,  # Most iterations (highly heterophilic)
        'finetune_enc_epochs': 20,  # Longer fine-tuning per iteration
        'finetune_disc_epochs': 15,  # Longer discriminator tuning
        'alpha': 5.0,
        'beta': 0.5,
        'lr_enc': 1.2e-3,  # Slightly higher LR
        'lr_disc': 1e-2,  # Slightly higher LR
    },
}


def get_config(dataset_name: str, **kwargs) -> Config:
    """
    Get configuration for a specific dataset with optional overrides.

    Args:
        dataset_name (str): Dataset name (cora, citeseer, pubmed, chameleon, squirrel, actor)
        **kwargs: Optional hyperparameter overrides

    Returns:
        Config: Configuration dataclass instance
    """
    dataset_lower = dataset_name.lower()

    if dataset_lower not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    # Create base config
    #change 1: changed to dataset_lower.
    cfg = Config(dataset=dataset_lower)

    # Apply dataset-specific overrides
    dataset_params = DATASET_CONFIGS[dataset_lower]
    for key, value in dataset_params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    # Apply user-provided overrides
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")

    return cfg


def print_config(cfg: Config) -> None:
    """print configuration."""
    print("=" * 60)
    print(f"Configuration for {cfg.dataset}")
    print("=" * 60)

    # Group related parameters
    groups = {
        'Dataset & Training': ['dataset', 'device', 'seed'],
        'Dimensions': ['in_dim', 'hidden_dim', 'out_dim', 'proj_dim'],
        'Augmentation': ['p_e', 'p_f'],
        'Contrastive Learning': ['tau', 'm'],
        'Discriminator': ['t'],
        'Conflict-based Weighting': ['alpha', 'beta', 'eta'],
        'Training Phases': ['pretrain_enc_epochs', 'pretrain_disc_epochs', 'n_iterations',
                            'finetune_enc_epochs', 'finetune_disc_epochs'],
        'Optimizers': ['lr_enc', 'lr_disc', 'dropout'],
        'Batch Processing': ['batch_size', 'k_nn'],
    }

    for group_name, keys in groups.items():
        print(f"\n{group_name}:")
        for key in keys:
            if hasattr(cfg, key):
                value = getattr(cfg, key)
                print(f"  {key}: {value}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration system...\n")

    # Test homophilic dataset
    print("Loading Cora config:")
    cfg_cora = get_config('cora')
    print(f"  tau={cfg_cora.tau}, m={cfg_cora.m}")

    # Test heterophilic dataset
    print("\nLoading Chameleon config:")
    cfg_chameleon = get_config('chameleon')
    print(f"  tau={cfg_chameleon.tau}, m={cfg_chameleon.m}")

    # Test with overrides
    print("\nLoading with custom tau:")
    cfg_custom = get_config('cora', tau=1.5, lr_enc=2e-3)
    print(f"  tau={cfg_custom.tau}, lr_enc={cfg_custom.lr_enc}")

    # Test pretty printing
    print("\n\nFull configuration print:")
    print_config(cfg_cora)

    print("\nConfiguration tests passed!")
