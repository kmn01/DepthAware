import ast
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class SceneEmbeddingDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.df["label_id"] = self.label_encoder.fit_transform(self.df["label"])

    def __len__(self):
        return len(self.df)

    def _parse_embedding(self, emb):
        if isinstance(emb, str):
            emb = ast.literal_eval(emb)
        return torch.tensor(emb, dtype=torch.float)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return {
            "image_emb": self._parse_embedding(row["image_embedding"]),
            "depth_emb": self._parse_embedding(row["depthmap_embedding"]),
            "text_emb": self._parse_embedding(row["caption_embedding"]),
            "label": torch.tensor(row["label_id"], dtype=torch.long),
        }

    @property
    def num_classes(self):
        return len(self.label_encoder.classes_)
