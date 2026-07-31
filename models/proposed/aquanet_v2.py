import torch
import torch.nn as nn
from .msrb import MSRB
from .csab import CSAB

class AquaNetV2(nn.Module):
    """
    AquaNet v2 Architecture (Hard-Masked Hierarchical Dual-Head Model)
    Custom Conv Stem + MSRB stages + CSAB attention + Dual Head (Binary Head + Type Head).
    Uses hard step-function masking: if b_prob >= 0.5 then type_pred else clean.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # MSRB + CSAB Stages
        self.stage1 = nn.Sequential(MSRB(64, 128), CSAB(128))
        self.stage2 = nn.Sequential(MSRB(128, 256), CSAB(256))
        self.stage3 = nn.Sequential(MSRB(256, 512), CSAB(512))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Shared Feature Representation
        self.shared_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Hierarchical Heads
        self.binary_head = nn.Linear(256, 1)  # 0 = Clean, 1 = Contaminated
        self.type_head = nn.Linear(256, 6)    # 6 Contamination Types

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        pooled = self.gap(x).view(x.size(0), -1)
        h = self.shared_fc(pooled)

        b_logits = self.binary_head(h)
        t_logits = self.type_head(h)

        return {
            'binary_logits': b_logits,
            'type_logits': t_logits
        }

    def predict_7class_hard(self, b_logits, t_logits):
        """Hard threshold prediction mapping to 7-class format."""
        b_prob = torch.sigmoid(b_logits.squeeze(1))
        t_pred = t_logits.argmax(dim=1)
        # Clean = 0; Contaminated Types = t_pred + 1
        flat_pred = torch.where(b_prob >= 0.5, t_pred + 1, torch.zeros_like(t_pred))
        return flat_pred
