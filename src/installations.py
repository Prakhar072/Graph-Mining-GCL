import torch
import torch_geometric
import numpy as np
import scipy
import sklearn
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import yaml
import tqdm

print("PyTorch:", torch.__version__)
print("PyG:", torch_geometric.__version__)
print("CUDA available:", torch.cuda.is_available())
print("All imports successful")