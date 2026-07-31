import os
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np

CLASSES = ['clean', 'algae', 'debris', 'foam', 'oil', 'turbid', 'uncertain']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class WaterQualityDataset(Dataset):
    """
    Dataset class supporting both flat 7-class targets and hierarchical dual-head targets.
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.samples = []
        self.labels = []
        
        split_dir = os.path.join(root_dir, split) if os.path.exists(os.path.join(root_dir, split)) else root_dir
        
        for cls in CLASSES:
            cls_idx = CLASS_TO_IDX[cls]
            p = os.path.join(split_dir, 'clean' if cls == 'clean' else f'contaminated/{cls}')
            if not os.path.exists(p):
                # Try direct subfolder
                p = os.path.join(split_dir, cls)
            if not os.path.exists(p):
                continue
                
            for fname in os.listdir(p):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fpath = os.path.join(p, fname)
                    self.samples.append((fpath, cls_idx))
                    self.labels.append(cls_idx)
                    
        print(f"[Dataset] Loaded {len(self.samples)} samples for split '{split}' from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        image = Image.open(fpath).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Hierarchical targets
        # Binary: 0 = Clean, 1 = Contaminated
        binary_label = 0 if label == 0 else 1
        # Type: 0..5 (Algae=0, Debris=1, Foam=2, Oil=3, Turbid=4, Uncertain=5)
        type_label = 0 if label == 0 else label - 1
        
        return image, label, binary_label, type_label

def get_weighted_sampler(dataset):
    """Create WeightedRandomSampler to balance class frequencies."""
    targets = np.array(dataset.labels)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / (class_counts + 1e-5)
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler
