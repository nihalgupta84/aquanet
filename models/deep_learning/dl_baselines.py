import torch
import torch.nn as nn
import timm

def get_dl_baseline_model(model_name, num_classes=7, pretrained=True):
    """
    Factory function to instantiate deep learning baseline models.
    Supported: 'resnet50', 'densenet121', 'efficientnet_b0', 'mobilenetv2_100'
    """
    model_name_map = {
        'resnet50': 'resnet50',
        'densenet121': 'densenet121',
        'efficientnet_b0': 'efficientnet_b0',
        'mobilenetv2': 'mobilenetv2_100'
    }

    timm_name = model_name_map.get(model_name.lower(), model_name)
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    return model
