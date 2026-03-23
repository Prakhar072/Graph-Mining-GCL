# Conflict-Conditioned Soft Contrastive Learning Framework for Graphs

A PyTorch implementation of a novel graph contrastive learning framework that adaptively weights structural and attribute-based supervision signals based on the graph's homophily-heterophily regime. This framework bridges homophilic and heterophilic graphs through a conflict index-driven weight calibration mechanism.

## Overview

This project implements a comprehensive framework for self-supervised learning on both homophilic and heterophilic graphs. The key innovation is the **Exponential Partitioned Similarity (SPART)** kernel combined with **conflict-conditioned soft contrastive learning** that dynamically adapts to the graph's structural properties.

### Key Features

- **SPART Similarity**: Exponential partitioned similarity that improves robustness to feature noise
- **Conflict Index**: Measures homophily-heterophily regime by comparing structural vs. attribute similarity
- **Soft Contrastive Loss**: Weighted supervision combining structural and attribute signals
- **Discriminator Calibration**: Fine-tuning with semantics-consistency discriminator for improved pair weighting
- **Multi-Phase Training**: Pre-training with contrastive learning + discriminator pre-training + fine-tuning iterations
- **Dataset Support**: Works with 6 major datasets (Cora, Citeseer, Pubmed, Chameleon, Squirrel, Actor)

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric 1.7+
- NumPy, Pandas, Matplotlib
- scikit-learn, scipy

### Setup

```bash
# Clone repository
git clone <repository_url>
cd Graph-Mining-GCL

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
Graph-Mining-GCL/
├── src/
│   ├── data.py              # [Not used - use dataset.py instead]
│   ├── dataset.py           # Dataset loading and preprocessing
│   ├── augment.py           # Edge dropping and feature masking
│   ├── encoder.py           # GCN encoder with projection head
│   ├── spart.py             # Exponential partitioned similarity
│   ├── conflict.py          # Conflict index computation
│   ├── weights.py           # Structural and attribute weight matrices
│   ├── loss.py              # Soft contrastive loss and weight combination
│   ├── discriminator.py     # Semantics-consistency discriminator
│   ├── train.py             # Full training pipeline
│   ├── evaluate.py          # Linear evaluation protocol
│   ├── config.py            # Hyperparameter management
│   └── installations.py     # Utilities
├── main.py                  # Entry point with CLI
├── requirements.txt         # Python dependencies
├── cache/                   # Preprocessed data cache (auto-created)
├── checkpoints/             # Model checkpoints (auto-created)
└── README.md               # This file
```

## Usage

### Quick Start

Run the complete pipeline on the Cora dataset:

```bash
# CPU
python main.py --dataset cora --device cpu

# GPU (if available)
python main.py --dataset cora --device cuda
```

### Command-Line Options

```bash
python main.py --help

Options:
  --dataset {cora,citeseer,pubmed,chameleon,squirrel,actor}
                        Dataset name (default: cora)
  --device {cpu,cuda}   Device to use (default: cpu)
  --seed SEED           Random seed for reproducibility (default: 42)
  --pretrain-epochs N   Override pre-training epochs
  --tau TAU             Override temperature parameter
  --evaluate            Run linear evaluation after training
  --n-eval-runs N       Number of evaluation runs (default: 10)
```

### Examples

```bash
# Custom hyperparameters
python main.py --dataset cora --pretrain-epochs 200 --tau 1.0

# Reproducible training with fixed seed
python main.py --dataset citeseer --seed 123
```

## Core Components

### 1. Dataset Loading (`dataset.py`)

Loads from PyTorch Geometric and applies preprocessing:
- **Adjacency Normalization**: D^{-1/2} * (A + I) * D^{-1/2}
- **Feature Normalization**: Row-wise L2 normalization
- **Caching**: Preprocessed data cached to `cache/` directory

**Datasets**:
- **Homophilic**: Cora, Citeseer, Pubmed
- **Heterophilic**: Chameleon, Squirrel, Actor

### 2. Graph Augmentation (`augment.py`)

Creates two augmented views per graph:
- **Edge Dropping**: Bernoulli sampling with probability p_e (default: 0.3)
- **Feature Masking**: Independent feature dimension dropout with probability p_f (default: 0.3)

### 3. SPART Similarity (`spart.py`)

Exponential partitioned similarity kernel:
```
S_SPART = (1/K) * sum_k exp(K * S_k / tau)
```
where each S_k is the similarity of a feature partition, providing robustness via Jensen's inequality.

### 4. Conflict Index (`conflict.py`)

Measures graph homophily-heterophily regime:
```
C = mean_pairs |PPR(i,j) - cos_sim(i,j)|
```
- **Structural Similarity**: Personalized PageRank (PPR) with α=0.15
- **Attribute Similarity**: Cosine similarity (dot product for row-normalized features)
- Computed on ~10K sampled pairs (half edges, half non-edges)

### 5. Weight Matrices (`weights.py`)

Generates supervision weights:
- **W_s**: Structural weights from PPR similarity
- **W_a**: Attribute weights from cosine similarity
- Both row-normalized (sum to 1 per row)

**Combined weights**:
```
W_total = (1 - lambda_C) * W_s + lambda_C * W_a
lambda_C = sigmoid(alpha * (C - beta))
```
where lambda_C adapts based on conflict index.

### 6. Soft Contrastive Loss (`loss.py`)

Weighted contrastive loss with soft targets:
```
L = -mean_i( sum_j W[i,j] * log_softmax(S[i,j]) )
```
where S is computed using SPART similarity and W provides adaptive weighting.

### 7. GCN Encoder (`encoder.py`)

Two-layer Graph Convolutional Network:
- **Architecture**: in_dim → hidden_dim → out_dim
- **Activation**: ReLU with dropout (0.1)
- **Projection Head**: 2-layer MLP for contrastive pre-training (discarded at evaluation)

### 8. Discriminator (`discriminator.py`)

Semantics-consistency discriminator for fine-tuning:
- **Architecture**: 3-layer MLP with LeakyReLU(0.2) and Sigmoid
- **Input Fusion**: Concatenates encoder output H with Laplacian eigenvectors
- **Pair Evaluation**: Hadamard product of fusion vectors
- **Loss**: Balanced softmax loss adjusting for class imbalance

## Training Pipeline

### Phase 1: Pre-Training Encoder (300 epochs)

```
1. Compute conflict index C from adjacency and features
2. Compute weight matrices W_s and W_a
3. Combine into W_total via conflict-adaptive sigmoid
4. For each epoch:
   - Sample augmented views (edge drop + feature mask)
   - Compute SPART similarity S
   - Apply soft contrastive loss with W_total
   - Update encoder + projection head
```

### Phase 2: Pre-Training Discriminator (25 epochs)

```
1. Extract frozen encoder representations H
2. Build fusion vectors Z = [H; eigenvectors]
3. Find positive pairs: nodes that are both connected AND k-NN similar
4. Sample negative pairs: non-edges
5. For each epoch:
   - Compute discriminator scores via Hadamard product
   - Compute balanced softmax loss
   - Update discriminator
```

### Phase 3: Fine-Tuning (20 iterations)

For each iteration:

**Step A - Encoder Update** (15 epochs):
```
- Sample augmented views
- Compute H_u, H_v (with gradients for encoder training)
- Build fusion vectors Z_u = [H_u; eigenvectors]
- Get discriminator scores (frozen discriminator, no_grad)
- Build calibrated weights: W_cali[i,j] = W_total[i,j] * 1(score >= eta)
- Apply soft contrastive loss with W_cali
- Update encoder only
```

**Step B - Discriminator Update** (10 epochs):
```
- Find new positive pairs in current H
- Build new fusion vectors
- Train discriminator with balanced softmax loss
- Update discriminator only
```

## Configuration

All hyperparameters are managed in `config.py`:

```python
# Basic usage
cfg = get_config('cora')  # Load dataset-specific config

# With overrides
cfg = get_config('cora', tau=1.0, lr_enc=5e-4)

# Key parameters
cfg.tau                # Temperature (0.8 for homophilic, 1.2 for heterophilic)
cfg.m                  # SPART partition size (128)
cfg.alpha, cfg.beta    # Sigmoid parameters for conflict weighting
cfg.eta                # Discriminator score threshold (0.6)
cfg.pretrain_enc_epochs  # 300
cfg.finetune_disc_epochs # 10
```

### Dataset-Specific Defaults

| Parameter | Cora | Citeseer | Pubmed | Chameleon | Squirrel | Actor |
|-----------|------|----------|--------|-----------|----------|-------|
| tau       | 0.8  | 0.8      | 0.8    | 1.2       | 1.2      | 1.2   |
| Pretrain  | 300  | 300      | 300    | 400       | 400      | 400   |

## Evaluation

### Linear Evaluation Protocol

After training, evaluate the encoder via linear probing:

```bash
python main.py --dataset cora --evaluate --n-eval-runs 10
```

The evaluation:
1. Extracts frozen encoder representations H
2. For 10 runs with random train/test splits:
   - Trains logistic regression on 10% training nodes
   - Evaluates accuracy on 90% test nodes
   - Records mean and standard deviation

### Results Interpretation

- **Mean Accuracy**: Average accuracy across runs
- **Std Dev**: Variability across random splits
- **Higher is Better**: Indicates better learned representations

## Reproducibility

All random seeds are set for reproducibility:

```python
# Automatically set in training
set_seed(42)  # Sets torch, numpy, and random seeds
```

Use `--seed` flag to set custom seed:
```bash
python main.py --dataset cora --seed 42 --evaluate
```

## Performance Tips

### Memory Efficiency

- **Reduce batch size** if OOM errors occur:
  ```bash
  python main.py --dataset chameleon --batch-size 512
  ```

- **Use CPU** for smaller graphs:
  ```bash
  python main.py --dataset cora --device cpu
  ```

### Training Speed

- **Fewer pre-training epochs**:
  ```bash
  python main.py --dataset cora --pretrain-epochs 100
  ```

- **Cached data**: First run downloads data; subsequent runs use cache from `cache/` directory

## Advanced Usage

### Custom Training Loop

```python
from src.train import train, set_seed
from src.evaluate import linear_evaluation

# Train
set_seed(42)
encoder, cfg = train('cora', device='cuda')

# Evaluate
results = linear_evaluation(encoder, X, edge_index, y, n_runs=10, device='cuda')
print(f"Accuracy: {results['mean_acc']:.4f} ± {results['std_acc']:.4f}")
```

### Loading Pre-trained Models

```python
import torch
from src.encoder import GCNEncoder

# Load checkpoint
encoder = GCNEncoder(in_dim=1433, hidden_dim=64, out_dim=256, dropout=0.1)
checkpoint = torch.load('checkpoints/encoder_pretrain_final.pt')
encoder.load_state_dict(checkpoint)
encoder.eval()
```

## Testing

Individual modules include built-in tests:

```bash
# Test SPART similarity
python src/spart.py

# Test soft contrastive loss
python src/loss.py

# Test dataset loading
python src/dataset.py

# Test all augmentation functions
python src/augment.py

# Test conflict index computation
python src/conflict.py

# Test encoder
python src/encoder.py

# Test configuration
python src/config.py

# Test evaluation
python src/evaluate.py
```
