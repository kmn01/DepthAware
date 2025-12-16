import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


@torch.no_grad()
def evaluate(model, loader, device, label_encoder=None):
    model.eval()

    all_preds = []
    all_labels = []

    for batch in loader:
        img = batch["image_emb"].to(device)
        depth = batch["depth_emb"].to(device)
        text = batch["text_emb"].to(device)
        labels = batch["label"].to(device)

        outputs = model(img, depth, text)
        preds = outputs.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=label_encoder.classes_ if label_encoder else None,
    )

    return report

def plot_confusion_matrix(cm, class_names=None, title=None):
    if class_names is not None:
        assert len(class_names) == cm.shape[0], (
            f"Confusion matrix has {cm.shape[0]} classes "
            f"but got {len(class_names)} labels"
        )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(xticks_rotation=45, cmap="Blues")

    if title:
        plt.title(title)
