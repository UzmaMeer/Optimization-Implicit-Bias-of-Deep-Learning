import torch

def compute_alpha_infinity(alpha_val, gamma, sgd_train_losses, X):
    """
    Computes alpha_infinity vector (Equation 5 / Section 5.3)
    """
    n = X.shape[0]
    
    # Approximate integral of the loss: \int L(beta_s) ds
    loss_integral = sum(sgd_train_losses) * gamma
    
    # Compute the diagonal of (X^T @ X) / n
    XTX_diag = torch.diag(X.T @ X) / n
    
    # Calculate alpha_infinity vector 
    alpha_inf = alpha_val * torch.exp(-2 * gamma * XTX_diag * loss_integral)
    return alpha_inf

def add_label_noise(y_i, delta_t):
    """
    Generate artificial label noise Delta_t (Section 5.4)
    Delta_t is randomly picked from {2*delta_t, -2*delta_t}
    """
    noise_sign = 1.0 if torch.rand(1) > 0.5 else -1.0
    Delta_t = noise_sign * (2 * delta_t)
    return Delta_t
