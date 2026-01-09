import torch
from utils import compute_f1


def training(model, train_loader, optimizer, criterion, device):
    model.to(device)

    model.train()
    total_loss = 0.0
    total = 0

    all_preds = []
    all_labels = []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs

        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach())
        all_labels.append(y.detach())

        avg_loss = total_loss / max(total, 1)

    return avg_loss


def evaluate(model, test_loader, criterion, device):
    model.to(device)
    model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            all_preds.append(preds)
            all_labels.append(y)

            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)

            correct += (preds == y).sum().item()
            total += y.size(0)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    f1 = compute_f1(all_labels, all_preds)

    result = {
        "loss": total_loss / total,
        "acc": correct / total,
        "f1": f1
    }

    return result
