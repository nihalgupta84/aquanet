import os, sys, argparse, json
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.metrics import compute_metrics
from utils.soft_gating import SoftProbabilisticGating
from dataset.transforms import get_transforms
from dataset.water_dataset import WaterQualityDataset, CLASSES
from models.deep_learning.dl_baselines import get_dl_baseline_model
from models.transformers.vision_transformers import get_transformer_model
from models.proposed.aquanet_v3 import AquaNetV3

def parse_args():
    parser = argparse.ArgumentParser(description="AquaNet Multi-Dataset Evaluation Script")
    parser.add_argument('--model', type=str, default='aquanet_v3', help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint .pth')
    parser.add_argument('--dataset', type=str, default='all', choices=['d1', 'd2', 'd3', 'all'], help='Dataset to evaluate on')
    return parser.parse_args()

def evaluate_dataset(model, data_dir, dataset_name, device):
    transform = get_transforms(224, is_train=False)
    ds = WaterQualityDataset(data_dir, split='test' if 'cleaned_water' in data_dir else '.', transform=transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    soft_gating = SoftProbabilisticGating().to(device)

    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, labels, b_labels, t_labels in loader:
            images = images.to(device)
            if hasattr(model, 'binary_head'):
                outputs = model(images)
                p_7class = soft_gating(outputs['binary_logits'], outputs['type_logits'])
                p = p_7class.argmax(dim=1)
            else:
                logits = model(images)
                p = logits.argmax(dim=1)

            preds.extend(p.cpu().numpy())
            targets.extend(labels.numpy())

    metrics = compute_metrics(targets, preds, CLASSES)
    print("\n" + "="*80)
    print(f" EVALUATION ON {dataset_name.upper()} ({len(ds)} IMAGES)")
    print("="*80)
    print(f"  7-Class Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro F1 Score:   {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
    return metrics

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'aquanet_v3':
        model = AquaNetV3(num_classes=7, pretrained=False).to(device)
    elif args.model in ['vit_tiny', 'swin_tiny']:
        model = get_transformer_model(args.model, num_classes=7, pretrained=False).to(device)
    else:
        model = get_dl_baseline_model(args.model, num_classes=7, pretrained=False).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint)
    print(f"Successfully loaded checkpoint from {args.checkpoint}")

    datasets_to_eval = {
        'Dataset 1 (Clean Benchmark)': './data/cleaned_water_dataset',
        'Dataset 2 (Scraped Real Finetune)': './data/cleaned_scrapper_finetune',
        'Dataset 3 (Unseen OOD Real Test)': './data/cleaned_scrapper_unseen_test'
    }

    results = {}
    for name, path in datasets_to_eval.items():
        if os.path.exists(path):
            results[name] = evaluate_dataset(model, path, name, device)

    os.makedirs('results', exist_ok=True)
    res_path = f"results/eval_{args.model}_all_datasets.json"
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAll dataset evaluation results saved to {res_path}")

if __name__ == '__main__':
    main()
