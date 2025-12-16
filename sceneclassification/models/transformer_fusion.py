import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerFusion(nn.Module):

    def __init__(
        self,
        img_dim,
        depth_dim,
        text_dim,
        hidden_dim,
        num_classes,
        num_layers=2,
        num_heads=4,
        dropout=0.2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Project each modality to hidden_dim with LayerNorm
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.depth_proj = nn.Sequential(
            nn.Linear(depth_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Learnable modality gates
        self.modality_gates = nn.Parameter(torch.ones(3))

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Modality embeddings: CLS + 3 modalities
        self.modality_embeddings = nn.Embedding(4, hidden_dim)
        # 0=CLS, 1=img, 2=depth, 3=text

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, img, depth, text):
        B = img.size(0)

        # Apply learnable modality gates
        gates = torch.softmax(self.modality_gates, dim=0)
        img = self.img_proj(img) * gates[0]
        depth = self.depth_proj(depth) * gates[1]
        text = self.text_proj(text) * gates[2]

        # Normalize embeddings
        img = F.normalize(img, dim=-1).unsqueeze(1)
        depth = F.normalize(depth, dim=-1).unsqueeze(1)
        text = F.normalize(text, dim=-1).unsqueeze(1)

        cls = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat([cls, img, depth, text], dim=1)
        modality_ids = torch.tensor([0, 1, 2, 3], device=tokens.device).unsqueeze(0)
        tokens = tokens + self.modality_embeddings(modality_ids)

        encoded = self.transformer(tokens)

        # Mean pooling over all tokens
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)

    def compute_gates(self, img, depth, text):
        """
        Return the current softmaxed modality gate weights.
        """
        return torch.softmax(self.modality_gates, dim=0).detach()