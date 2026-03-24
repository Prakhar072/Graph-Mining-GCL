# Conflict-Aware Soft Contrastive Learning on Graphs

A PyTorch implementation of a novel graph contrastive learning framework that adaptively weights structural and attribute-based supervision signals based on the graph's homophily-heterophily regime. This framework bridges homophilic and heterophilic graphs through a conflict index-driven weight calibration mechanism.

### Quick Start

Run the complete pipeline on the Cora dataset:

```bash
# CPU
python main.py --dataset cora --device cpu

# GPU (if available)
python main.py --dataset cora --device cuda
```


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
