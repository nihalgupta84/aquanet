import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftProbabilisticGating(nn.Module):
    """
    Differentiable Soft Probabilistic Hierarchical Dual-Head Gating.
    
    P(Clean) = P_head1(Clean)
    P(Class_k) = P_head1(Contaminated) * P_head2(Type_k)  for k in {1..6}
    
    Guarantees sum(P_7class) = 1.0 and propagates gradients to both heads simultaneously.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, binary_logits, type_logits):
        """
        Args:
            binary_logits: (B, 1) or (B, 2)
            type_logits: (B, 6)
        Returns:
            p_7class: (B, 7) joint probability distribution
        """
        if binary_logits.dim() == 2 and binary_logits.size(1) == 1:
            p_contam = torch.sigmoid(binary_logits / self.temperature)
            p_clean = 1.0 - p_contam
        else:
            b_probs = F.softmax(binary_logits / self.temperature, dim=1)
            p_clean = b_probs[:, 0:1]
            p_contam = b_probs[:, 1:2]

        p_type = F.softmax(type_logits / self.temperature, dim=1)

        # Joint 7-class probability: [Clean, Algae, Debris, Foam, Oil, Turbid, Uncertain]
        p_7class = torch.cat([p_clean, p_contam * p_type], dim=1)
        return p_7class
