import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_preds, total_labels = [], []
    start_time = time.time()

    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)

        preds = out.argmax(dim=1)
        total_preds.extend(preds.cpu().numpy())
        total_labels.extend(y.cpu().numpy())
        
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    end_time = time.time()

    acc = accuracy_score(total_labels, total_preds)
    print(f"  [Train] Avg Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | Time: {end_time - start_time:.2f}s")
    return avg_loss


def evaluate(model, dataloader, device, val=False, return_preds=False):
    model.eval()
    total_loss = 0
    total_preds, total_labels = [], []
    start_time = time.time()

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)

            preds = out.argmax(dim=1)
            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(y.cpu().numpy())

            loss = F.cross_entropy(out, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    end_time = time.time()

    acc = accuracy_score(total_labels, total_preds)
    if val:
        print(f"  [Eval] Avg Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | Time: {end_time - start_time:.2f}s")

    if return_preds:
        return avg_loss, acc, total_preds, total_labels
    return avg_loss, acc