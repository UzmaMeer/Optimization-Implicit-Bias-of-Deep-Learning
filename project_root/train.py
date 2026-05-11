import os
import yaml
import torch
import json
from src.dataset import generate_synthetic_data, save_data
from src.model import initialize_weights, get_beta, calculate_loss, calculate_test_loss
from src.utils import compute_alpha_infinity, add_label_noise

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_gd(X, y, beta_star, d, alpha_val, gamma, n_iterations):
    print("Starting GD Training...")
    w_plus, w_minus = initialize_weights(d, alpha_val)
    train_losses, test_losses = [], []
    
    for t in range(n_iterations):
        if w_plus.grad is not None: w_plus.grad.zero_()
        if w_minus.grad is not None: w_minus.grad.zero_()
            
        beta = get_beta(w_plus, w_minus)
        loss = calculate_loss(beta, X, y)
        test_loss = calculate_test_loss(beta, beta_star)
        
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        
        loss.backward()
        with torch.no_grad():
            w_plus -= gamma * w_plus.grad
            w_minus -= gamma * w_minus.grad
            
        if t % 10000 == 0:
            print(f"GD Iter {t}: Train = {loss.item():.2e}, Test = {test_loss.item():.4f}")
            
    return train_losses, test_losses

def run_sgd(X, y, beta_star, d, alpha_val, gamma, n_iterations):
    print("Starting SGD Training (Batch Size = 1)...")
    n = X.shape[0]
    w_plus, w_minus = initialize_weights(d, alpha_val)
    train_losses, test_losses = [], []
    
    for t in range(n_iterations):
        idx = torch.randint(0, n, (1,))
        x_i, y_i = X[idx], y[idx]
        
        beta = get_beta(w_plus, w_minus)
        sample_prediction = x_i @ beta
        sample_loss = (sample_prediction - y_i)**2 / 4 
        
        with torch.no_grad():
            train_losses.append(calculate_loss(beta, X, y).item())
            test_losses.append(calculate_test_loss(beta, beta_star).item())
            
        if w_plus.grad is not None: w_plus.grad.zero_()
        if w_minus.grad is not None: w_minus.grad.zero_()
        sample_loss.backward()
        
        with torch.no_grad():
            w_plus -= gamma * w_plus.grad
            w_minus -= gamma * w_minus.grad
            
        if t % 10000 == 0:
            print(f"SGD Iter {t}: Full Train = {train_losses[-1]:.2e}, Test = {test_losses[-1]:.4f}")
            
    return train_losses, test_losses

def run_gd_inf(X, y, beta_star, alpha_inf, gamma, n_iterations):
    print("Starting GD Training from alpha_inf...")
    w_plus = alpha_inf.clone().detach().requires_grad_(True)
    w_minus = alpha_inf.clone().detach().requires_grad_(True)
    train_losses, test_losses = [], []
    
    for t in range(n_iterations):
        if w_plus.grad is not None: w_plus.grad.zero_()
        if w_minus.grad is not None: w_minus.grad.zero_()
            
        beta = get_beta(w_plus, w_minus)
        loss = calculate_loss(beta, X, y)
        test_loss = calculate_test_loss(beta, beta_star)
        
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        
        loss.backward()
        with torch.no_grad():
            w_plus -= gamma * w_plus.grad
            w_minus -= gamma * w_minus.grad
            
        if t % 10000 == 0:
            print(f"GD_INF Iter {t}: Train = {loss.item():.2e}, Test = {test_loss.item():.4f}")
            
    return train_losses, test_losses

def run_sgd_noise(X, y, beta_star, d, alpha_val, gamma, delta_t, n_iterations):
    print(f"Starting SGD Training with Label Noise (delta = {delta_t})...")
    n = X.shape[0]
    w_plus, w_minus = initialize_weights(d, alpha_val)
    train_losses, test_losses = [], []
    
    for t in range(n_iterations):
        idx = torch.randint(0, n, (1,))
        x_i, y_i = X[idx], y[idx]
        
        Delta_t = add_label_noise(y_i, delta_t)
        beta = get_beta(w_plus, w_minus)
        
        sample_prediction = x_i @ beta
        diff = sample_prediction - y_i - Delta_t
        perturbed_loss = (diff**2) / 4
        
        with torch.no_grad():
            train_losses.append(calculate_loss(beta, X, y).item())
            test_losses.append(calculate_test_loss(beta, beta_star).item())
            
        if w_plus.grad is not None: w_plus.grad.zero_()
        if w_minus.grad is not None: w_minus.grad.zero_()
        perturbed_loss.backward()
        
        with torch.no_grad():
            w_plus -= gamma * w_plus.grad
            w_minus -= gamma * w_minus.grad
            
        if t % 10000 == 0:
            print(f"SGD_NOISE Iter {t}: Full Train = {train_losses[-1]:.2e}, Test = {test_losses[-1]:.4f}")
            
    return train_losses, test_losses

def main():
    config = load_config()
    dc = config["data"]
    hc = config["hyperparameters"]
    pc = config["paths"]
    
    os.makedirs(pc["results_dir"], exist_ok=True)
    os.makedirs(pc["checkpoints_dir"], exist_ok=True)
    
    # 1. Setup Data
    X, y, beta_star, indices = generate_synthetic_data(dc["n_samples"], dc["d_features"], dc["k_sparsity"], hc["seed"])
    save_data(X, y, beta_star, pc["data_dir"])
    
    # 2. Run Trainings
    gd_tr, gd_te = run_gd(X, y, beta_star, dc["d_features"], hc["alpha_val"], hc["gamma"], hc["n_iterations"])
    sgd_tr, sgd_te = run_sgd(X, y, beta_star, dc["d_features"], hc["alpha_val"], hc["gamma"], hc["n_iterations"])
    
    alpha_inf = compute_alpha_infinity(hc["alpha_val"], hc["gamma"], sgd_tr, X)
    gd_inf_tr, gd_inf_te = run_gd_inf(X, y, beta_star, alpha_inf, hc["gamma"], hc["n_iterations"])
    
    sgd_n_tr, sgd_n_te = run_sgd_noise(X, y, beta_star, dc["d_features"], hc["alpha_val"], hc["gamma"], hc["delta_t"], hc["n_iterations"])
    
    # 3. Save Results
    results = {
        "gd": {"train": gd_tr, "test": gd_te},
        "sgd": {"train": sgd_tr, "test": sgd_te},
        "gd_inf": {"train": gd_inf_tr, "test": gd_inf_te},
        "sgd_noise": {"train": sgd_n_tr, "test": sgd_n_te}
    }
    
    with open(os.path.join(pc["results_dir"], "training_history.json"), "w") as f:
        json.dump(results, f)
        
    print("Training complete! Logs saved to results/training_history.json")

if __name__ == "__main__":
    main()
