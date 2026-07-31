import torch
import torch.nn as nn
import timm

def get_transformer_model(model_name, num_classes=7, pretrained=True):
    """
    Factory function for Vision Transformers.
    Supported: 'vit_tiny', 'swin_tiny'
    """
    timm_map = {
        'vit_tiny': 'vit_tiny_patch16_224',
        'swin_tiny': 'swin_tiny_patch4_window7_224'
    }

    timm_name = timm_map.get(model_name.lower(), model_name)
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    return model
