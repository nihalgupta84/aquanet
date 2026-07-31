import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .msrb import MSRB
from .csab import CSAB

# 7 Domain-Specific Natural Language Class Text Prompts
WATER_TEXT_PROMPTS = [
    "clear transparent water body with natural riverbed or subtle surface ripple",
    "green algae bloom, duckweed, or floating photosynthetic surface scum",
    "floating solid plastic waste, trash, organic twigs, or litter in water",
    "white soapy froth, thick surface foam, or organic bubbling scum",
    "oily slick, iridescent rainbow sheen, or petroleum film floating on water",
    "muddy brown turbid water with high suspended sediment concentration",
    "ambiguous contaminated water surface with mixed organic and chemical waste"
]

class ContrastiveReasoningModule(nn.Module):
    """
    Contrastive Reasoning Module (Anchor-Guided Cross-Attention)
    Computes cross-attention between spatial visual tokens F(i, j) and 7 Text Anchors A_text.
    Produces a 7-channel semantic guidance heatmap S that anchors visual features.
    """
    def __init__(self, visual_dim, text_dim=512, proj_dim=256):
        super().__init__()
        self.visual_proj = nn.Conv2d(visual_dim, proj_dim, kernel_size=1)
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.scale = proj_dim ** -0.5

    def forward(self, visual_features, text_embeds):
        """
        Args:
            visual_features: (B, C, H, W)
            text_embeds: (7, D_text) or (B, 7, D_text)
        Returns:
            semantic_heatmap: (B, 7, H, W)
            anchored_features: (B, C + 7, H, W)
        """
        B, C, H, W = visual_features.shape
        v_proj = self.visual_proj(visual_features) # (B, proj_dim, H, W)
        v_flat = v_proj.flatten(2).transpose(1, 2)  # (B, H*W, proj_dim)
        
        if text_embeds.dim() == 2:
            t_proj = self.text_proj(text_embeds).unsqueeze(0).repeat(B, 1, 1) # (B, 7, proj_dim)
        else:
            t_proj = self.text_proj(text_embeds)
            
        # Cosine similarity cross-attention heatmap
        v_norm = F.normalize(v_flat, dim=-1)
        t_norm = F.normalize(t_proj, dim=-1)
        
        # (B, H*W, proj_dim) x (B, proj_dim, 7) -> (B, H*W, 7)
        sim_matrix = torch.bmm(v_norm, t_norm.transpose(1, 2)) * self.scale
        heatmap = sim_matrix.transpose(1, 2).view(B, 7, H, W) # (B, 7, H, W)
        
        # Concatenate semantic guidance heatmap to visual features
        anchored_features = torch.cat([visual_features, heatmap], dim=1) # (B, C+7, H, W)
        return heatmap, anchored_features

class AquaNetVLM(nn.Module):
    """
    AquaNet-VLM Architecture
    
    1. Pretrained VLM Vision Backbone (EVA02 / CLIP / ViT / ConvNeXt-CLIP)
    2. Text-Guided Class Anchors & Contrastive Reasoning Module
    3. MSRB + CSAB Feature Enhancement Neck (Multi-scale fluid extraction & glare attention)
    4. Soft Probabilistic Hierarchical Dual-Head
    """
    def __init__(self, num_classes=7, pretrained=True, vlm_backbone='eva02_base_patch14_clip_224'):
        super().__init__()
        # Try loading specified CLIP backbone from timm or fallback to vit_base_patch16_clip_224
        try:
            self.backbone = timm.create_model(vlm_backbone, pretrained=pretrained, num_classes=0)
        except Exception:
            self.backbone = timm.create_model('vit_base_patch16_clip_224', pretrained=pretrained, num_classes=0)
            
        # Feature dimension
        in_features = getattr(self.backbone, 'num_features', 768)
        
        # Learnable / Encoded Text Anchor Matrix (7 classes x 512 dim)
        self.text_anchors = nn.Parameter(torch.randn(7, 512) * 0.02)
        
        # Contrastive Reasoning Module
        self.contrastive_reasoning = ContrastiveReasoningModule(visual_dim=in_features, text_dim=512, proj_dim=256)
        
        # MSRB + CSAB Neck operating on anchored features (in_features + 7 channels)
        self.msrb = MSRB(in_channels=in_features + 7, out_channels=512)
        self.csab = CSAB(channels=512)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Shared Feature Representation
        self.shared_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Hierarchical Heads
        self.binary_head = nn.Linear(256, 1)   # 0 = Clean, 1 = Contaminated
        self.type_head = nn.Linear(256, 6)     # 6 Contamination Types
        self.flat_head = nn.Linear(256, num_classes)

    def forward(self, x):
        B = x.size(0)
        
        # 1. Visual Feature Extraction
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
            
        if features.dim() == 3: # (B, N, C) -> transform to spatial grid (B, C, H, W)
            B, N, C = features.shape
            H = W = int(N ** 0.5)
            if H * W == N:
                features = features.transpose(1, 2).view(B, C, H, W)
            else: # Exclude CLS token if present
                features = features[:, 1:, :].transpose(1, 2).view(B, C, int((N-1)**0.5), int((N-1)**0.5))

        # 2. Contrastive Reasoning Module (Anchor-Guided Cross Attention)
        heatmap, anchored_features = self.contrastive_reasoning(features, self.text_anchors)
        
        # 3. MSRB + CSAB Neck
        msrb_out = self.msrb(anchored_features)
        csab_out = self.csab(msrb_out)
        
        # 4. Global Pooling & Shared FC
        pooled = self.gap(csab_out).view(B, -1)
        h = self.shared_fc(pooled)
        
        # 5. Output Heads
        b_logits = self.binary_head(h)
        t_logits = self.type_head(h)
        flat_logits = self.flat_head(h)
        
        return {
            'binary_logits': b_logits,
            'type_logits': t_logits,
            'flat_logits': flat_logits,
            'semantic_heatmap': heatmap
        }
