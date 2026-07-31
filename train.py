import os, sys, argparse, time, yaml, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.seed import set_seed
from utils.logger import setup_logger
from utils.metrics import compute_metrics
from utils.soft_gating import SoftProbabilisticGating
from dataset.transforms import get_transforms
from dataset.water_dataset import WaterQualityDataset, get_weighted_sampler, CLASSES
from models.deep_learning.dl_baselines import get_dl_baseline_model
from models.transformers.vision_transformers import get_transformer_model
from models.proposed.aquanet_v3 import AquaNetV3

from models.proposed.aquanet_v1 import AquaNetV1
from models.proposed.aquanet_v2 import AquaNetV2
from models.proposed.aquanet_v3 import AquaNetV3
from models.proposed.aquanet_vlm import AquaNetVLM

import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="AquaNet Water Quality Classification Training")
    parser.add_argument('--model', type=str, default='aquanet_vlm',
                        choices=['aquanet_v1', 'aquanet_v2', 'aquanet_v3', 'aquanet_vlm', 'densenet121', 'resnet50', 'efficientnet_b0', 'mobilenetv2', 'vit_tiny', 'swin_tiny'],
                        help='Model architecture to train')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    return parser.parse_args()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def main():
    args = parse_args()
    set_seed(args.seed)

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger = setup_logger(name="aquanet_train", log_dir=config['checkpoint']['log_dir'], log_file=f"train_{args.model}.log")
    logger.info(f"Starting Training | Model: {args.model} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # W&B Login / Setup
    secrets_file = "/workspace/.secrets/api_keys.env"
    if os.path.exists(secrets_file):
        with open(secrets_file) as f:
            for line in f:
                if line.startswith("WANDB_API_KEY="):
                    os.environ['WANDB_API_KEY'] = line.split("=", 1)[1].strip()

    if args.wandb and os.environ.get('WANDB_API_KEY'):
        wandb.init(project=config['wandb']['project'], name=f"{args.model}-clean-training", config=vars(args))
    else:
        os.environ['WANDB_MODE'] = 'offline'
        wandb.init(project=config['wandb']['project'], mode='offline')

    # Data Loaders
    data_dir = config['dataset']['root_dir']
    train_transform = get_transforms(config['dataset']['image_size'], is_train=True)
    val_transform = get_transforms(config['dataset']['image_size'], is_train=False)

    train_ds = WaterQualityDataset(data_dir, split='train', transform=train_transform)
    val_ds = WaterQualityDataset(data_dir, split='val', transform=val_transform)
    test_ds = WaterQualityDataset(data_dir, split='test', transform=val_transform)

    train_sampler = get_weighted_sampler(train_ds) if config['training']['use_weighted_sampler'] else None
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=config['training']['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=config['training']['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=config['training']['num_workers'])

    # Instantiate Model
    if args.model == 'aquanet_v1':
        model = AquaNetV1(num_classes=7).to(device)
    elif args.model == 'aquanet_v2':
        model = AquaNetV2(num_classes=7).to(device)
    elif args.model == 'aquanet_v3':
        model = AquaNetV3(num_classes=7, pretrained=True).to(device)
    elif args.model == 'aquanet_vlm':
        model = AquaNetVLM(num_classes=7, pretrained=True).to(device)
    elif args.model in ['vit_tiny', 'swin_tiny']:
        model = get_transformer_model(args.model, num_classes=7, pretrained=True).to(device)
    else:
        model = get_dl_baseline_model(args.model, num_classes=7, pretrained=True).to(device)

    # Class Weights for Focal Loss
    targets = np.array(train_ds.labels)
    class_counts = np.bincount(targets, minlength=7)
    weights = torch.tensor(1.0 / (class_counts + 1e-5), dtype=torch.float32).to(device)
    weights = weights / weights.sum()

    criterion = FocalLoss(alpha=weights, gamma=config['training']['focal_gamma']) if config['training']['use_focal_loss'] else nn.CrossEntropyLoss(weight=weights)
    soft_gating = SoftProbabilisticGating().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['training']['scheduler_factor'], patience=config['training']['scheduler_patience'])

    best_val_loss = float('inf')
    best_val_acc = 0.0
    save_path = os.path.join(config['checkpoint']['save_dir'], f"{args.model}_best.pth")
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()

        for images, labels, b_labels, t_labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if args.model in ['aquanet_v3', 'aquanet_vlm']:
                outputs = model(images)
                p_7class = soft_gating(outputs['binary_logits'], outputs['type_logits'])
                loss_joint = nn.functional.nll_loss(torch.log(p_7class + 1e-7), labels, weight=weights)
                loss_flat = criterion(outputs['flat_logits'], labels)
                loss = 0.5 * loss_joint + 0.5 * loss_flat
                preds = p_7class.argmax(dim=1)
            elif args.model == 'aquanet_v2':
                outputs = model(images)
                b_loss = nn.functional.binary_cross_entropy_with_logits(outputs['binary_logits'].squeeze(1), (labels > 0).float())
                contam_mask = (labels > 0)
                t_loss = nn.functional.cross_entropy(outputs['type_logits'][contam_mask], labels[contam_mask] - 1) if contam_mask.sum() > 0 else 0.0
                loss = b_loss + t_loss
                preds = model.predict_7class_hard(outputs['binary_logits'], outputs['type_logits'])
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for images, labels, b_labels, t_labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                if args.model in ['aquanet_v3', 'aquanet_vlm']:
                    outputs = model(images)
                    p_7class = soft_gating(outputs['binary_logits'], outputs['type_logits'])
                    loss_joint = nn.functional.nll_loss(torch.log(p_7class + 1e-7), labels, weight=weights)
                    loss_flat = criterion(outputs['flat_logits'], labels)
                    loss = 0.5 * loss_joint + 0.5 * loss_flat
                    preds = p_7class.argmax(dim=1)
                elif args.model == 'aquanet_v2':
                    outputs = model(images)
                    b_loss = nn.functional.binary_cross_entropy_with_logits(outputs['binary_logits'].squeeze(1), (labels > 0).float())
                    contam_mask = (labels > 0)
                    t_loss = nn.functional.cross_entropy(outputs['type_logits'][contam_mask], labels[contam_mask] - 1) if contam_mask.sum() > 0 else 0.0
                    loss = b_loss + t_loss
                    preds = model.predict_7class_hard(outputs['binary_logits'], outputs['type_logits'])
                else:
                    logits = model(images)
                    loss = criterion(logits, labels)
                    preds = logits.argmax(dim=1)

                val_loss_sum += loss.item() * images.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss = val_loss_sum / len(val_ds)
        val_metrics = compute_metrics(val_targets, val_preds, CLASSES)
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['macro_f1']

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        logger.info(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | Time: {elapsed:.1f}s")

        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'val/macro_f1': val_f1
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'val_acc': val_acc, 'val_f1': val_f1}, save_path)
            logger.info(f"  --> Saved best model checkpoint to {save_path} (Val Acc: {val_acc:.4f})")

    logger.info(f"\nTraining Complete! Best Val Acc: {best_val_acc:.4f} | Best Val Loss: {best_val_loss:.4f}")

    # Final Evaluation on Test Set
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for images, labels, b_labels, t_labels in test_loader:
            images = images.to(device)
            if args.model in ['aquanet_v3', 'aquanet_vlm']:
                outputs = model(images)
                p_7class = soft_gating(outputs['binary_logits'], outputs['type_logits'])
                preds = p_7class.argmax(dim=1)
            elif args.model == 'aquanet_v2':
                outputs = model(images)
                preds = model.predict_7class_hard(outputs['binary_logits'], outputs['type_logits'])
            else:
                logits = model(images)
                preds = logits.argmax(dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.numpy())

    test_metrics = compute_metrics(test_targets, test_preds, CLASSES)
    logger.info("\n" + "="*80)
    logger.info(f" FINAL TEST RESULTS ({args.model.upper()}) ON CLEAN LEAKAGE-FREE DATASET")
    logger.info("="*80)
    logger.info(f"  Test 7-Class Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    logger.info(f"  Test Macro F1 Score:   {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Test Weighted F1:      {test_metrics['weighted_f1']:.4f}")

    # Save results JSON
    os.makedirs(config['checkpoint']['results_dir'], exist_ok=True)
    res_file = os.path.join(config['checkpoint']['results_dir'], f"test_results_{args.model}.json")
    with open(res_file, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Results saved to {res_file}")

    wandb.finish()

if __name__ == '__main__':
    main()
