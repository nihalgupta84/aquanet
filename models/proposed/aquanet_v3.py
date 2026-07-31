import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .msrb import MSRB
from .csab import CSAB

class AquaNetV3(nn.Module):
    """
    AquaNet v3 Proposed Architecture
    
    1. Pretrained DenseNet121 Backbone Foundation (Preserves dense channels & visual priors)
    2. MSRB + CSAB Feature Enhancement Neck (Multi-scale fluid extraction & glare suppression)
    3. Hierarchical Dual-Head with Soft Probabilistic Gating
    """
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        # Load DenseNet121 backbone features
        self.backbone = timm.create_model('densenet121', pretrained=pretrained, num_classes=0)
        
        # DenseNet121 feature channel dimension is 1024
        in_features = 1024
        
        # MSRB + CSAB Neck
        self.msrb = MSRB(in_channels=in_features, out_channels=512)
        self.csab = CSAB(channels=512)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Shared Representation
        self.shared_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Hierarchical Output Heads
        self.binary_head = nn.Linear(256, 1)    # 0 = Clean, 1 = Contaminated
        self.type_head = nn.Linear(256, 6)      # 6 Contamination types
        
        # Direct 7-class head (for joint soft gating)
        self.flat_head = nn.Linear(256, num_classes)

    def forward(self, x):
        # 1. Backbone dense feature maps
        features = self.backbone.forward_features(x)  # (B, 1024, H, W)
        
        # 2. MSRB + CSAB Neck
        msrb_out = self.msrb(features)                # (B, 512, H, W)
        csab_out = self.csab(msrb_out)                # (B, 512, H, W)
        
        # 3. Global Pooling & Shared FC
        pooled = self.gap(csab_out).view(csab_out.size(0), -1)
        h = self.shared_fc(pooled)
        
        # 4. Heads
        b_logits = self.binary_head(h)
        t_logits = self.type_head(h)
        flat_logits = self.flat_head(h)
        
        return {
            'binary_logits': b_logits,
            'type_logits': t_logits,
            'flat_logits': flat_logits
        }
