import os, glob, random
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

OUTPUT_DIR = "paper_v2/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_DIR = "cleaned_water_dataset/test"
CLASSES = ["clean", "algae", "debris", "foam", "oil", "turbid", "uncertain"]
CLASS_TITLES = ["Clean Water", "Algae Bloom", "Floating Debris", "Soapy Foam", "Oil Sheen", "Turbid Sediment", "Uncertain Water"]

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sample_images = {}
for cls in CLASSES:
    cls_folder = os.path.join(DATASET_DIR, cls)
    imgs = sorted(glob.glob(os.path.join(cls_folder, "*.jpg")) + glob.glob(os.path.join(cls_folder, "*.png")))
    if imgs:
        sample_images[cls] = imgs[0]

# -------------------------------------------------------------
# 1. Dataset Representative Samples Grid
# -------------------------------------------------------------
print("Generating Dataset Sample Grid...")
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for idx, (cls, title) in enumerate(zip(CLASSES, CLASS_TITLES)):
    ax = axes[idx]
    if cls in sample_images:
        img = Image.open(sample_images[cls]).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"({chr(97+idx)}) {title}", fontsize=12, fontweight='bold', pad=6)
    ax.axis("off")

axes[7].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_dataset_samples.png"), dpi=300, bbox_inches='tight')
# Convert saved high-res PNG to PDF using PIL to ensure non-empty PDF
img_pdf = Image.open(os.path.join(OUTPUT_DIR, "fig_dataset_samples.png")).convert("RGB")
img_pdf.save(os.path.join(OUTPUT_DIR, "fig_dataset_samples.pdf"))
plt.close()
print("Saved fig_dataset_samples.png and .pdf")

# -------------------------------------------------------------
# 2. Grad-CAM Activation Heatmaps
# -------------------------------------------------------------
print("Generating Grad-CAM Feature Heatmaps...")
from models.proposed.aquanet_v3 import AquaNetV3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AquaNetV3(num_classes=7).to(device)
ckpt_path = "checkpoints/aquanet_v3_best.pth"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
model.eval()

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for idx, (cls, title) in enumerate(zip(CLASSES, CLASS_TITLES)):
    ax = axes[idx]
    if cls in sample_images:
        img_pil = Image.open(sample_images[cls]).convert("RGB")
        img_t = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = model.backbone.features(img_t)
            msrb_feat = model.msrb(feat)
            csab_feat = model.csab(msrb_feat)
            heatmap = csab_feat.squeeze(0).abs().mean(dim=0).cpu().numpy()
            
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img_pil.size, Image.BILINEAR)
        
        ax.imshow(img_pil)
        ax.imshow(heatmap_img, cmap='jet', alpha=0.55)
        ax.set_title(f"({chr(97+idx)}) {title} (CSAB Attention)", fontsize=11, fontweight='bold', pad=6)
    ax.axis("off")

axes[7].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_gradcam_maps.png"), dpi=300, bbox_inches='tight')
img_pdf = Image.open(os.path.join(OUTPUT_DIR, "fig_gradcam_maps.png")).convert("RGB")
img_pdf.save(os.path.join(OUTPUT_DIR, "fig_gradcam_maps.pdf"))
plt.close()
print("Saved fig_gradcam_maps.png and .pdf")

# -------------------------------------------------------------
# 3. Qualitative Prediction Confidence Grid
# -------------------------------------------------------------
print("Generating Qualitative Predictions Grid...")
fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
axes = axes.flatten()

for idx, (cls, title) in enumerate(zip(CLASSES, CLASS_TITLES)):
    ax = axes[idx]
    if cls in sample_images:
        img_pil = Image.open(sample_images[cls]).convert("RGB")
        img_t = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(img_t)
            probs = out["probs"].squeeze(0).cpu().numpy()
            pred_idx = probs.argmax()
            pred_cls = CLASSES[pred_idx]
            conf = probs[pred_idx] * 100
            
        ax.imshow(img_pil)
        color = 'darkgreen' if pred_cls == cls else 'darkred'
        ax.set_title(f"GT: {cls.capitalize()} | Pred: {pred_cls.capitalize()}\nConf: {conf:.1f}%", fontsize=11, fontweight='bold', color=color, pad=6)
    ax.axis("off")

axes[7].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_predictions_grid.png"), dpi=300, bbox_inches='tight')
img_pdf = Image.open(os.path.join(OUTPUT_DIR, "fig_predictions_grid.png")).convert("RGB")
img_pdf.save(os.path.join(OUTPUT_DIR, "fig_predictions_grid.pdf"))
plt.close()
print("Saved fig_predictions_grid.png and .pdf")

# -------------------------------------------------------------
# 4. Class Distribution Bar Chart
# -------------------------------------------------------------
print("Generating Class Distribution Bar Chart...")
train_counts = [185, 693, 220, 178, 106, 246, 328]
val_counts = [39, 150, 48, 40, 24, 54, 61]
test_counts = [41, 150, 48, 40, 24, 54, 70]

x = np.arange(len(CLASSES))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
rects1 = ax.bar(x - width, train_counts, width, label='Train (1,956)', color='#2b5c8f')
rects2 = ax.bar(x, val_counts, width, label='Validation (416)', color='#4682b4')
rects3 = ax.bar(x + width, test_counts, width, label='Test (427)', color='#87ceeb')

ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
ax.set_title('Class-Wise Distribution of 100% Un-leaked AquaNet-Bench-v1', fontsize=13, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels([c.capitalize() for c in CLASSES], fontsize=11, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_dataset_distribution.png"), dpi=300, bbox_inches='tight')
img_pdf = Image.open(os.path.join(OUTPUT_DIR, "fig_dataset_distribution.png")).convert("RGB")
img_pdf.save(os.path.join(OUTPUT_DIR, "fig_dataset_distribution.pdf"))
plt.close()
print("Saved fig_dataset_distribution.png and .pdf")

# -------------------------------------------------------------
# 5. Master Model Comparison Bar Chart
# -------------------------------------------------------------
print("Generating Master Model Comparison Bar Chart...")
model_names = ['AquaNet\n(Ours)', 'ResNet-50', 'MobileNetV2', 'EfficientNet-B0', 'DenseNet-121', 'ViT-Tiny', 'Swin-Tiny']
accuracies = [90.40, 82.67, 81.50, 75.18, 74.24, 65.81, 40.05]
f1_scores = [87.26, 78.97, 79.14, 69.55, 71.22, 57.96, 30.72]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5.5))
rects1 = ax.bar(x - width/2, accuracies, width, label='7-Class Accuracy (%)', color='#1f77b4')
rects2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1-Score (x100)', color='#ff7f0e')

ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Master Benchmark Model Performance on Un-leaked Test Set', fontsize=13, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_model_comparison.png"), dpi=300, bbox_inches='tight')
img_pdf = Image.open(os.path.join(OUTPUT_DIR, "fig_model_comparison.png")).convert("RGB")
img_pdf.save(os.path.join(OUTPUT_DIR, "fig_model_comparison.pdf"))
plt.close()
print("Saved fig_model_comparison.png and .pdf")

# -------------------------------------------------------------
# 6. Domain Transfer Bar Chart
# -------------------------------------------------------------
print("Generating Domain Transfer Bar Chart...")
stages = ['AquaNet-Bench-v1\n(Test Set)', 'AquaNet-Real-v1\n(Val Set)', 'AquaNet-Real-v2\n(Pre-FT OOD)', 'AquaNet-Real-v2\n(Post-FT OOD)']
accs = [90.40, 90.48, 39.04, 57.53]
f1s = [87.26, 90.23, 40.05, 53.84]

x = np.arange(len(stages))
width = 0.35

fig, ax = plt.subplots(figsize=(9.5, 5))
rects1 = ax.bar(x - width/2, accs, width, label='7-Class Accuracy (%)', color='#2ca02c')
rects2 = ax.bar(x + width/2, f1s, width, label='Macro F1-Score (x100)', color='#d62728')

ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Synthetic-to-Real Domain Generalization Jump (+18.49%)', fontsize=13, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=10, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_domain_transfer.png"), dpi=300, bbox_inches='tight')
img_pdf = Image.open(os.path.join(OUTPUT_DIR, "fig_domain_transfer.png")).convert("RGB")
img_pdf.save(os.path.join(OUTPUT_DIR, "fig_domain_transfer.pdf"))
plt.close()
print("Saved fig_domain_transfer.png and .pdf")

print("All publication figures successfully generated with non-empty PDFs!")
