import os, sys, argparse, json
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.transforms import get_transforms
from dataset.water_dataset import CLASSES
from utils.soft_gating import SoftProbabilisticGating
from models.proposed.aquanet_v3 import AquaNetV3
from models.deep_learning.dl_baselines import get_dl_baseline_model

def predict_image(image_path, model, device):
    """Predict water quality condition and contamination type for a single image."""
    transform = get_transforms(224, is_train=False)
    image = Image.open(image_path).convert('RGB')
    tensor_img = transform(image).unsqueeze(0).to(device)

    soft_gating = SoftProbabilisticGating().to(device)
    model.eval()

    with torch.no_grad():
        if hasattr(model, 'binary_head'):
            outputs = model(tensor_img)
            p_7class = soft_gating(outputs['binary_logits'], outputs['type_logits'])[0]
            probs = p_7class.cpu().numpy()
        else:
            logits = model(tensor_img)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    is_clean = (pred_class == 'clean')

    return {
        'image_path': image_path,
        'prediction': pred_class,
        'confidence': confidence,
        'is_clean': is_clean,
        'class_probabilities': {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    }

def main():
    parser = argparse.ArgumentParser(description="AquaNet Water Quality Single Image Inference")
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='aquanet_v3', help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'aquanet_v3':
        model = AquaNetV3(num_classes=7, pretrained=False).to(device)
    else:
        model = get_dl_baseline_model(args.model, num_classes=7, pretrained=False).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint)

    res = predict_image(args.image, model, device)
    print("\n" + "="*60)
    print(" AQUANET INFERENCE RESULT")
    print("="*60)
    print(f" Image:       {res['image_path']}")
    print(f" Prediction:  {res['prediction'].upper()}")
    print(f" Confidence:  {res['confidence'] * 100:.2f}%")
    print(f" Status:      {'CLEAN WATER' if res['is_clean'] else 'CONTAMINATED WATER'}")
    print("\nClass Probabilities:")
    for k, v in res['class_probabilities'].items():
        print(f"  {k:<12}: {v*100:6.2f}%")

if __name__ == '__main__':
    main()
