import torch
import numpy as np
import os

def generate_synthetic_data(n=40, d=100, k=5, seed=42):
    """
    Generates the sparse ground truth beta* and features X.
    Returns: X, y, beta_star, indices
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate a sparse ground truth beta*
    beta_star = torch.zeros(d)
    indices = torch.randperm(d)[:k]
    beta_star[indices] = torch.sign(torch.randn(k)) # Random +1 or -1 values
    
    # Generate features X ~ N(0, I) and labels y = X @ beta_star
    X = torch.randn(n, d)
    y = X @ beta_star
    
    return X, y, beta_star, indices

def save_data(X, y, beta_star, save_dir="data"):
    """Saves the generated tensors to disk."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(X, os.path.join(save_dir, "X.pt"))
    torch.save(y, os.path.join(save_dir, "y.pt"))
    torch.save(beta_star, os.path.join(save_dir, "beta_star.pt"))

def load_data(load_dir="data"):
    """Loads the generated tensors from disk."""
    X = torch.load(os.path.join(load_dir, "X.pt"))
    y = torch.load(os.path.join(load_dir, "y.pt"))
    beta_star = torch.load(os.path.join(load_dir, "beta_star.pt"))
    return X, y, beta_star
