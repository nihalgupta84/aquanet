import os, sys, argparse, time, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.seed import set_seed
from utils.logger import setup_logger
from utils.metrics import compute_metrics
from utils.soft_gating import SoftProbabilisticGating
from dataset.transforms import get_transforms
from dataset.water_dataset import WaterQualityDataset, get_weighted_sampler, CLASSES
from models.proposed.aquanet_v3 import AquaNetV3

def parse_args():
    parser = argparse.ArgumentParser(description="AquaNet Real-World Fine-Tuning on Dataset 2")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/aquanet_v3_best.pth', help='Pretrained checkpoint from Dataset 1')
    parser.add_argument('--epochs', type=int, default=20, help='Fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for fine-tuning')
    parser.add_argument('--lr', type=float, default=0.0001, help='Fine-tuning learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    logger = setup_logger(name="aquanet_finetune", log_dir="./logs", log_file="finetune_real_data.log")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Starting Fine-Tuning on Real Video Dataset 2 | Batch Size: {args.batch_size} | LR: {args.lr}")

    # Load Model & Checkpoint
    model = AquaNetV3(num_classes=7, pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint)
    logger.info(f"Loaded Dataset 1 pretrained checkpoint from {args.checkpoint}")

    # Load Dataset 2 (cleaned_scrapper_finetune - 213 images)
    data_dir_d2 = "./data/cleaned_scrapper_finetune"
    train_transform = get_transforms(224, is_train=True)
    val_transform = get_transforms(224, is_train=False)

    full_ds_train = WaterQualityDataset(data_dir_d2, split='.', transform=train_transform)
    full_ds_val = WaterQualityDataset(data_dir_d2, split='.', transform=val_transform)

    # 80/20 Train/Val Split for Fine-Tuning
    n_samples = len(full_ds_train)
    val_size = max(int(0.2 * n_samples), 1)
    train_size = n_samples - val_size

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, _ = random_split(full_ds_train, [train_size, val_size], generator=generator)
    _, val_ds = random_split(full_ds_val, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    soft_gating = SoftProbabilisticGating().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    save_path = "./checkpoints/aquanet_v3_finetuned_d2.pth"

    # Fine-Tuning Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()

        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            p_7class = soft_gating(outputs['binary_logits'], outputs['type_logits'])
            loss = nn.functional.nll_loss(torch.log(p_7class + 1e-7), labels)
            loss.backward()
            optimizer.step()

            preds = p_7class.argmax(dim=1)
            train_loss_sum += loss.item() * images.size(0)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                p_7class = soft_gating(outputs['binary_logits'], outputs['type_logits'])
                preds = p_7class.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_metrics = compute_metrics(val_targets, val_preds, CLASSES)
        val_acc = val_metrics['accuracy']
        elapsed = time.time() - t0

        logger.info(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} F1: {val_metrics['macro_f1']:.4f} | Time: {elapsed:.1f}s")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'val_acc': val_acc}, save_path)

    logger.info(f"\nFine-Tuning Complete! Best Val Acc: {best_val_acc * 100:.2f}% | Saved to {save_path}")

    # Evaluate Fine-Tuned Model on Dataset 3 (Unseen Real OOD Test Set)
    data_dir_d3 = "./data/cleaned_scrapper_unseen_test"
    d3_ds = WaterQualityDataset(data_dir_d3, split='.', transform=val_transform)
    d3_loader = DataLoader(d3_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    finetuned_checkpoint = torch.load(save_path)
    model.load_state_dict(finetuned_checkpoint['model_state'])
    model.eval()

    d3_preds, d3_targets = [], []
    with torch.no_grad():
        for images, labels, _, _ in d3_loader:
            images = images.to(device)
            outputs = model(images)
            p_7class = soft_gating(outputs['binary_logits'], outputs['type_logits'])
            preds = p_7class.argmax(dim=1)
            d3_preds.extend(preds.cpu().numpy())
            d3_targets.extend(labels.numpy())

    d3_metrics = compute_metrics(d3_targets, d3_preds, CLASSES)
    logger.info("\n" + "="*80)
    logger.info(f" EVALUATION ON DATASET 3 (UNSEEN OOD REAL TEST) AFTER REAL FINE-TUNING")
    logger.info("="*80)
    logger.info(f"  Fine-Tuned 7-Class Accuracy: {d3_metrics['accuracy'] * 100:.2f}%")
    logger.info(f"  Fine-Tuned Macro F1 Score:   {d3_metrics['macro_f1']:.4f}")
    logger.info(f"  Fine-Tuned Weighted F1:      {d3_metrics['weighted_f1']:.4f}")

    os.makedirs('results', exist_ok=True)
    res_path = "results/eval_aquanet_v3_finetuned_d3.json"
    with open(res_path, 'w') as f:
        json.dump(d3_metrics, f, indent=2)
    logger.info(f"Fine-tuned OOD results saved to {res_path}")

if __name__ == '__main__':
    main()
