import torch
import time
from models.proposed.aquanet_v3 import AquaNetV3
import torchvision.models as models
from torchvision.models import resnet50, mobilenet_v2, efficientnet_b0, densenet121

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(1, 3, 224, 224).to(device)

model_list = [
    ("AquaNet (Proposed)", AquaNetV3(num_classes=7)),
    ("ResNet-50", resnet50(num_classes=7)),
    ("MobileNetV2", mobilenet_v2(num_classes=7)),
    ("EfficientNet-B0", efficientnet_b0(num_classes=7)),
    ("DenseNet-121", densenet121(num_classes=7)),
]

print(f"{'Model':<22} | {'Params (M)':<10} | {'Memory (MB)':<12} | {'GPU Latency (ms)':<16} | {'FPS':<8}")
print("-" * 75)

for name, model in model_list:
    model = model.to(device)
    model.eval()
    
    # Param count
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # Memory estimation
    mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    mem_mb = mem_bytes / (1024 * 1024)
    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
            
    # CUDA Latency benchmark
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    end_event.record()
    torch.cuda.synchronize()
    
    avg_latency = start_event.elapsed_time(end_event) / 100.0  # ms
    fps = 1000.0 / avg_latency
    
    print(f"{name:<22} | {params:<10.2f} | {mem_mb:<12.2f} | {avg_latency:<16.2f} | {fps:<8.1f}")
