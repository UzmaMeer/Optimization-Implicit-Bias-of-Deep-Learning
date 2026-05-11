import torch

def get_beta(w_plus, w_minus):
    """Computes beta = w+^2 - w-^2 as per Equation in Section 2.1"""
    return w_plus**2 - w_minus**2

def calculate_loss(beta, X, y):
    """
    Function to calculate training loss
    Mean Squared Error / 4n as per paper eq 1
    """
    n_samples = X.shape[0]
    predictions = X @ beta
    return torch.sum((predictions - y)**2) / (4 * n_samples)

def calculate_test_loss(beta, beta_star):
    """
    Function to calculate validation loss 
    (l2 distance to beta_star)
    """
    return torch.norm(beta - beta_star)**2

def initialize_weights(d, alpha_val):
    """
    Initialize weights w+ and w- as vectors of size d
    Using alpha as the initial value for both ensures beta_init = 0
    """
    w_plus = torch.full((d,), alpha_val, requires_grad=True)
    w_minus = torch.full((d,), alpha_val, requires_grad=True)
    return w_plus, w_minus
