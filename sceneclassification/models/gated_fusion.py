import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    def __init__(self, img_dim, depth_dim, text_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()

        self.img_gate = nn.Linear(img_dim, img_dim)
        self.depth_gate = nn.Linear(depth_dim, depth_dim)
        self.text_gate = nn.Linear(text_dim, text_dim)

        self.classifier = nn.Sequential(
            nn.Linear(img_dim + depth_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, img, depth, text):
        g_img = torch.sigmoid(self.img_gate(img))
        g_depth = torch.sigmoid(self.depth_gate(depth))
        g_text = torch.sigmoid(self.text_gate(text))

        fused = torch.cat(
            [g_img * img, g_depth * depth, g_text * text],
            dim=1
        )
        fused = F.normalize(fused, dim=1)

        return self.classifier(fused)
    
    def compute_gates(self, img, depth, text):
        """
        Returns the gate weights for each modality before fusion
        """
        img_proj = self.img_gate(img)
        depth_proj = self.depth_gate(depth)
        text_proj = self.text_gate(text)

        gates = torch.stack([img_proj, depth_proj, text_proj], dim=1)  # [B, 3]
        gates = torch.softmax(gates, dim=1)
        return gates
