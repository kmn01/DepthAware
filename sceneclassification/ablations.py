import torch
import numpy as np
from sklearn.metrics import confusion_matrix


ABLATION_CONFIGS = {
    "image_only": ["depth", "text"],
    "depth_only": ["image", "text"],
    "text_only": ["image", "depth"],
    "image_depth": ["text"],
    "image_text": ["depth"],
    "depth_text": ["image"],
    "all_modalities": [],
}


def run_ablation(batch, ablate_modalities, device, normalize=True):
    """
    Use specified modalities only, zero out others
    """
    img = batch["image_emb"].to(device)
    depth = batch["depth_emb"].to(device)
    text = batch["text_emb"].to(device)

    if "image" in ablate_modalities:
        img = torch.zeros_like(img)
    if "depth" in ablate_modalities:
        depth = torch.zeros_like(depth)
    if "text" in ablate_modalities:
        text = torch.zeros_like(text)

    return img, depth, text


@torch.no_grad()
def evaluate_with_ablation(
    model,
    dataloader,
    device,
    ablate_modalities=None,
):
    """
    Returns accuracy, confusion matrix, y_true, y_pred
    """
    model.eval()
    ablate_modalities = ablate_modalities or []

    y_true, y_pred = [], []

    for batch in dataloader:
        img, depth, text = run_ablation(batch, ablate_modalities, device)
        labels = batch["label"].to(device)

        outputs = model(img, depth, text)
        preds = outputs.argmax(dim=1)

        y_true.append(labels.cpu())
        y_pred.append(preds.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    acc = (y_true == y_pred).mean()
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }
