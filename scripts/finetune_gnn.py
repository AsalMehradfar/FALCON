import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.circuit_gnn import CircuitGNN
from utils.io_tools import load_yaml, seed_everything
from train.train_eval_gnn import train, evaluate
from utils.visual_utils import plot_loss_curves
from utils.model_utils import init_weights


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


root_dir = "./dataset" 
data_config = load_yaml('./config/data_config.yaml')
global_perf_list = list(data_config['Performance'].keys())
circuit_to_code = data_config['Classes']

batch_size = 256
num_epochs = 1000
patience = 7
lr = 1e-4
hidden_dim = 256

# Load dataset and wrap in DataLoader
gnn_data = torch.load(f"{root_dir}/gnn_heldout_data.pt", weights_only=False)

train_data = gnn_data["train"]
val_data = gnn_data["val"]
# train_data = Subset(gnn_data["train"], range(5000))
# val_data = Subset(gnn_data["val"], range(100))
print(f"[INFO] {len(train_data)} train samples, {len(val_data)} validation samples.")

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

param_templates = load_yaml('./Dataset/param_templates.yaml')
str_params_templates = load_yaml('./Dataset/str_params_templates.yaml')

# ==== Model ====
model = CircuitGNN(
    hidden_dim=hidden_dim,
    out_dim=len(global_perf_list),
    param_templates=param_templates,
    str_params_templates=str_params_templates
).to(device)

model.apply(init_weights)
pretrained_path = "./checkpoints/best_gnn_model.pt"
if pretrained_path:
    print(f"[INFO] Loading pretrained weights from {pretrained_path}")
    model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
# else:
#     model.apply(init_weights)

# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the final output MLP
for param in model.output_mlp.parameters():
    param.requires_grad = True

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
)

# ==== Training Loop ====
best_val_loss = float("inf")
# best_val_loss = evaluate(model, val_loader, device)
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    print(f"\n🎯 Epoch {epoch}/{num_epochs}")

    train_loss = train(model, train_loader, optimizer, device)
    val_loss = evaluate(model, val_loader, device)
    scheduler.step(val_loss)

    print(f"  🟢 Train Loss: {train_loss:.4f}")
    print(f"  🔵 Val   Loss: {val_loss:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "./checkpoints/best_gnn_model_finetuned.pt")
        print("  💾 Saved best model.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("  ⏹️ Early stopping triggered.")
            break

plot_loss_curves(train_losses, val_losses, title="GNN Training Loss", save_path='./plots/gnn/GNNLossFinetuned.pdf', log_scale=False)
