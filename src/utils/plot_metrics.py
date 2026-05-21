import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_results(json_path, output_path):
    """Read metrics.json file created during training to convert the scores into plots."""
    
    file_path = Path(json_path)
    if not file_path.exists():
        print(f"Error: Could not find '{json_path}'. Please check the path.")
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Load training data
    with open(file_path, "r") as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        print("Error: No history found in the JSON file.")
        return

    epochs = [epoch["epoch"] for epoch in history]
    losses = [epoch["loss"] for epoch in history]
    bleu4 = [epoch["bleu4"] for epoch in history]
    rougeL = [epoch["rougeL"] for epoch in history]
    cider = [epoch["cider"] for epoch in history]
    meteor = [epoch["meteor"] for epoch in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 1. Plot training loss
    ax1.plot(epochs, losses, marker='o', color='#d62728', linewidth=2, label='Train Loss')
    ax1.set_title("Model Convergence (Training Loss)", fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(epochs) 
    ax1.legend()

    # 2. Plot evaluation metrics
    ax2.plot(epochs, rougeL, marker='s', color="#FF0000", linewidth=2, label='ROUGE-L')
    ax2.plot(epochs, meteor, marker='^', color="#11ff00", linewidth=2, label='METEOR')
    ax2.plot(epochs, cider, marker='D', color="#ff8c00", linewidth=2, label='CIDEr')
    ax2.plot(epochs, bleu4, marker='v', color="#000dff", linewidth=2, label='BLEU-4')

    ax2.set_title("Caption quality evaluation on validation test", fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_xticks(epochs)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Success! Saved plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from training metrics.")
    parser.add_argument(
        "--json_path", 
        type=str, 
        required=True, 
        help="Path to the metrics json file to plot"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="src/utils/plots/training_plot.png", 
        help="path including name of plot (default: src/utils/plots/training_plot.png)"
    )
    
    args = parser.parse_args()
    plot_training_results(args.json_path, args.output)

