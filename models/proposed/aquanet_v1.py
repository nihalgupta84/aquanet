import torch
import torch.nn as nn
from .msrb import MSRB
from .csab import CSAB

class AquaNetV1(nn.Module):
    """
    AquaNet v1 Architecture (Flat Single-Head Model)
    Custom Conv stem + 4 MSRB stages + CSAB attention blocks + Single Flat 7-Class Head.
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

        # Flat 7-Class Head
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        pooled = self.gap(x).view(x.size(0), -1)
        logits = self.head(pooled)
        return logits
