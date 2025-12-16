import torch
import torch.nn as nn
import torch.nn.functional as F

class LateFusionShallow(nn.Module):
    def __init__(
            self, 
            img_dim, 
            depth_dim, 
            text_dim, 
            hidden_dim, 
            num_classes, 
            dropout=0.3
        ):
        super().__init__()
        input_dim = img_dim + depth_dim + text_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, img, depth, text):
        fused = torch.cat([img, depth, text], dim=1)
        fused = F.normalize(fused, dim=1)
        return self.fc(fused)
