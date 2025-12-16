import torch
from tqdm import tqdm
from sceneclassification.utils import accuracy


def train_epoch(model, dataloader, optimizer, criterion, device, ablate_modalities=[]):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in dataloader:
        optimizer.zero_grad()

        img = batch['image_emb'].to(device)
        depth = batch['depth_emb'].to(device)
        text = batch['text_emb'].to(device)

        if 'image' in ablate_modalities:
            img = torch.zeros_like(img)
        if 'depth' in ablate_modalities:
            depth = torch.zeros_like(depth)
        if 'text' in ablate_modalities:
            text = torch.zeros_like(text)

        outputs = model(img, depth, text)
        labels = batch['label'].to(device)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total



@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for batch in loader:
        img = batch["image_emb"].to(device)
        depth = batch["depth_emb"].to(device)
        text = batch["text_emb"].to(device)
        labels = batch["label"].to(device)

        outputs = model(img, depth, text)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        total_acc += accuracy(outputs, labels)

    return total_loss / len(loader), total_acc / len(loader)
