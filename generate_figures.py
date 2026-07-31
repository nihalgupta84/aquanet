import os, sys, json, glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set aesthetic publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

CLASSES = ['Clean', 'Algae', 'Debris', 'Foam', 'Oil', 'Turbid', 'Uncertain']
FIG_DIR = "paper_v1/figures"
os.makedirs(FIG_DIR, exist_ok=True)

def generate_confusion_matrix_plot():
    v3_json = "results/test_results_aquanet_v3.json"
    if not os.path.exists(v3_json):
        print(f"File {v3_json} not found.")
        return

    with open(v3_json) as f:
        data = json.load(f)

    cm = np.array(data['confusion_matrix'])
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=True)
    
    ax.set_title("AquaNet v3 Confusion Matrix (Un-leaked Test Set)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Class", fontsize=11, fontweight='bold')
    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, "fig_confusion_matrix.png")
    out_pdf = os.path.join(FIG_DIR, "fig_confusion_matrix.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved confusion matrix figure to {out_png}")

def generate_model_comparison_plot():
    models = ['AquaNet v3', 'AquaNet v2', 'ResNet50', 'MobileNetV2', 'EfficientNet-B0', 'DenseNet121', 'AquaNet-VLM', 'ViT-Tiny', 'AquaNet v1', 'Swin-Tiny']
    accs = [90.40, 83.37, 82.67, 81.50, 75.18, 74.24, 72.83, 65.81, 61.36, 40.05]
    f1s = [0.8726, 0.7857, 0.7897, 0.7914, 0.6955, 0.7122, 0.6753, 0.5796, 0.5248, 0.3072]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = '#1f77b4' # Deep Blue
    color2 = '#ff7f0e' # Vibrant Orange

    rects1 = ax1.bar(x - width/2, accs, width, label='Test Accuracy (%)', color=color1, alpha=0.85)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, [f*100 for f in f1s], width, label='Macro F1 (x100)', color=color2, alpha=0.85)

    ax1.set_ylabel('Test Accuracy (%)', color=color1, fontweight='bold')
    ax2.set_ylabel('Macro F1 Score (x100)', color=color2, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha='right', fontweight='bold')
    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 105)

    plt.title("Comparative Evaluation Across 10 Model Architectures (100% Un-leaked Dataset)", fontsize=13, fontweight='bold', pad=12)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, "fig_model_comparison.png")
    out_pdf = os.path.join(FIG_DIR, "fig_model_comparison.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved model comparison figure to {out_png}")

def generate_domain_transfer_plot():
    stages = ['D1 Clean Benchmark', 'D2 Fine-Tune Val', 'D3 Zero-Shot (Pre-FT)', 'D3 Unseen Test (Post-FT)']
    accs = [90.40, 90.48, 39.04, 57.53]
    colors = ['#2ca02c', '#1f77b4', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(stages, accs, color=colors, width=0.5, alpha=0.9)

    for bar, acc in zip(bars, accs):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{acc:.2f}%", ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel("7-Class Accuracy (%)", fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_title("AquaNet v3 Synthetic-to-Real Domain Generalization", fontsize=12, fontweight='bold', pad=12)
    plt.xticks(fontweight='bold', rotation=15)
    plt.tight_layout()

    out_png = os.path.join(FIG_DIR, "fig_domain_transfer.png")
    out_pdf = os.path.join(FIG_DIR, "fig_domain_transfer.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved domain transfer figure to {out_png}")

if __name__ == '__main__':
    generate_confusion_matrix_plot()
    generate_model_comparison_plot()
    generate_domain_transfer_plot()
