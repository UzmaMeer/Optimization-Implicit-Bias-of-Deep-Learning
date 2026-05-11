import os
import json
import matplotlib.pyplot as plt

def load_results(filepath="results/training_history.json"):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Please run train.py first.")
        return None
    with open(filepath, "r") as f:
        return json.load(f)

def plot_comparison(results, output_dir="results"):
    print("Generating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    gd = results["gd"]
    sgd = results["sgd"]
    gd_inf = results["gd_inf"]
    sgd_noise = results["sgd_noise"]
    
    plt.figure(figsize=(14, 6))

    # Subplot 1: Test Loss Comparison (All Models)
    plt.plot(gd["test"], label=r'GD from $\alpha$ (Baseline)', color='black', linewidth=2)
    plt.plot(sgd["test"], label=r'Standard SGD', color='blue', linestyle='--', alpha=0.8)
    plt.plot(gd_inf["test"], label=r'GD from $\alpha_\infty$ (Theoretical)', color='red', alpha=0.8)
    plt.plot(sgd_noise["test"], label=r'SGD + Label Noise (Doped)', color='green', linestyle='-.')

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Iteration t')
    plt.ylabel(r'Test loss $||\beta_t - \beta^*||_2^2$')
    plt.title('The Benefit of Stochasticity: Generalization Comparison')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plot_path = os.path.join(output_dir, "generalization_comparison.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    # plt.show() # Uncomment to display in an interactive environment
    
def print_final_metrics(results):
    print("\n--- Final Reproduction Results ---")
    print(f"1. Baseline GD Test Loss:        {results['gd']['test'][-1]:.6f}")
    print(f"2. Standard SGD Test Loss:       {results['sgd']['test'][-1]:.6f}")
    print(f"3. GD from alpha_inf Test Loss:  {results['gd_inf']['test'][-1]:.6f}")
    print(f"4. SGD + Label Noise Test Loss:  {results['sgd_noise']['test'][-1]:.6f}")

def main():
    results = load_results()
    if results:
        plot_comparison(results)
        print_final_metrics(results)

if __name__ == "__main__":
    main()
