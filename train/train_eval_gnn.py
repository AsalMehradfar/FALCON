import torch
import torch.nn.functional as F
from tqdm import tqdm

from train.loss import masked_mse_loss


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)
        # print("🔹 y_perf:", batch.y_normalized.shape)        # should be [B, 16]
        # print("🔹 mask:  ", batch.performance_mask.shape)     # should be same
        optimizer.zero_grad()

        out = model(batch)  # [batch_size, num_performance_metrics]
        loss = masked_mse_loss(out, batch.y_normalized, batch.performance_mask)
        loss.backward()
        # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # print(f"Clipped Gradient Norm: {total_norm:.4f}")
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            out = model(batch)
            loss = masked_mse_loss(out, batch.y_normalized, batch.performance_mask)
            total_loss += loss.item()

    return total_loss / len(dataloader)
