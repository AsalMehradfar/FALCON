import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from torch_geometric.loader import DataLoader
from collections import Counter
import numpy as np
import torch
from copy import deepcopy

from models.mlp_classifier import MLPClassifier
from train.train_eval_mlp import train, evaluate
from utils.io_tools import load_yaml, seed_everything
from utils.visual_utils import plot_loss_curves


YELLOW = "\033[93m"
RESET = "\033[0m"
seed_everything(42)

# -------------------------------
# Config
# -------------------------------
data_config = load_yaml('./config/data_config.yaml')
global_perf_list = list(data_config['Performance'].keys())
circuit_to_code = data_config['Classes']

root_dir = "dataset"
batch_size = 256
hidden_dim = 256
epochs = 200
lr = 0.0001

# -------------------------------
# Load Splitted Dataset
# -------------------------------
print("[INFO] Loading Splitted dataset...")

data = torch.load("./dataset/performance_class_data.pt", weights_only=True)
train_dataset = data["train"]
val_dataset = data["val"]
test_dataset = data["test"]
print("[INFO] Loaded cached dataset.")

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

train_labels = [y for _, y in train_dataset]
val_labels = [y for _, y in val_dataset]
test_labels = [y for _, y in test_dataset]
print(f"{YELLOW}[DEBUG] Train class distribution:", dict(sorted(Counter(train_labels).items())), f"{RESET}")
print(f"{YELLOW}[DEBUG] Val class distribution:", dict(sorted(Counter(val_labels).items())), f"{RESET}")
print(f"{YELLOW}[DEBUG] Test class distribution:", dict(sorted(Counter(test_labels).items())), f"{RESET}")

# -------------------------------
# Model Setup
# -------------------------------
print("[INFO] Creating model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(global_perf_list)
output_dim = len(circuit_to_code)

model = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -------------------------------
# Training Loop
# -------------------------------
print("[INFO] Starting training...")

best_model = deepcopy(model)
best_loss = np.inf
train_losses = []
val_losses = []

for epoch in range(epochs):
    print(f"\n[INFO] Epoch {epoch+1}/{epochs}")
    train_loss = train(model, train_loader, optimizer, device)
    val_loss, _ = evaluate(model, val_loader, device, val=True)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_loss:
        best_model = deepcopy(model)
        best_loss = val_loss

torch.save(best_model.state_dict(), "checkpoints/best_mlp_model.pt")
print("💾 Saved best model.")
plot_loss_curves(train_losses, val_losses, title="MLP Training Loss", save_path='./plots/mlp/MLPLoss.pdf', log_scale=False)

test_loss, test_acc, test_preds, test_labels = evaluate(best_model, test_loader, device, return_preds=True)
print(f"[RESULT] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")