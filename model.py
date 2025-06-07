import torch
import torch.nn as nn
from torchvision import models

class EfficientNetModel(nn.Module):
    def __init__(self):
        super(EfficientNetModel, self).__init__()
        
        # Pretrained EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Replace the final classifier layer for 20-class output
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 20)
        
    def forward(self, x):
        return self.backbone(x)
