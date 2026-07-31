import torch
import torch.nn as nn

class CSAB(nn.Module):
    """
    Cross Spatial-Channel Attention Block (CSAB)
    Refines channel importance and spatial focus to suppress water surface glare & sky reflections.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_ch = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, channels, kernel_size=1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Channel Attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        c_att = self.sigmoid_channel(avg_out + max_out)
        x_c = x * c_att

        # 2. Spatial Attention
        avg_spatial = torch.mean(x_c, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_c, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_spatial, max_spatial], dim=1)
        s_att = self.spatial_conv(spatial_in)

        out = x_c * s_att
        return out
