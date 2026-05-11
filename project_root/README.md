# Optimization & Implicit Bias of Deep Learning

This repository reproduces key experiments investigating the implicit bias of Stochastic Gradient Descent (SGD) compared to Gradient Descent (GD) in deep learning, based on established research. 

## Project Structure

```
project-root/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml               # Hyperparameter configurations
├── train.py                  # Main script for training models (GD, SGD, Label Noise)
├── inference.py              # Script for visualization and reporting metrics
├── data/                     # Synthetic data folder
├── notebooks/                # Exploratory Jupyter Notebooks
│   └── 01_implicit_bias_demo.ipynb
├── src/                      # Source modules
│   ├── dataset.py            # Data generation logic
│   ├── model.py              # Parameterization and loss functions
│   └── utils.py              # Mathematical helper functions
├── results/                  # Plots and training metrics
└── checkpoints/              # Model states and weights
```

## Installation

Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training Simulations
To run the simulations (which generates synthetic data, trains GD, SGD, computes alpha_infinity, and runs SGD with Label Noise), run:
```bash
python train.py
```
This will compute the implicit bias trajectories and save the history to `results/training_history.json`.

### 2. Evaluation & Visualization
To generate the convergence and generalization error plots demonstrating the benefit of stochasticity, run:
```bash
python inference.py
```
The comparison plots will be saved in the `results/` folder, and the final numeric metrics will be printed to the console.
