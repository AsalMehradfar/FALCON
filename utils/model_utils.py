import torch
from torch_geometric.loader import DataLoader
import numpy as np

from models.circuit_gnn import CircuitGNN  # adjust path
from utils.io_tools import load_yaml


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif hasattr(m, 'lin') and isinstance(m.lin, torch.nn.Linear):
        # For GCNConv, which has internal .lin layers
        torch.nn.init.xavier_uniform_(m.lin.weight)
        if m.lin.bias is not None:
            torch.nn.init.zeros_(m.lin.bias)


def load_model(device, perf_dim=16):

    # Load trained model
    param_templates = load_yaml('./dataset/param_templates.yaml')
    str_params_templates = load_yaml('./dataset/str_params_templates.yaml')
    hidden_dim = 256

    model = CircuitGNN(
    hidden_dim=hidden_dim,
    out_dim=perf_dim,
    param_templates=param_templates,
    str_params_templates=str_params_templates
    ).to(device)
    model.load_state_dict(torch.load("./checkpoints/best_gnn_model.pt", map_location="cpu", weights_only=True))
    model.eval()

    return model


def load_data(loader=True, heldout=True):

    # Load dataset and wrap in DataLoader
    root_dir = "dataset"
    test_dataset = torch.load(f"{root_dir}/gnn_test_data.pt", weights_only=False)["test"]
    scalers = test_dataset.scaler

    config = load_yaml("./config/data_config.yaml")
    global_perf_dict = config["Performance"]

    if heldout:
        heldout_dataset = torch.load(f"{root_dir}/gnn_heldout_data.pt", weights_only=False)["test"]

    if loader:
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        if heldout:
            heldout_loader = DataLoader(heldout_dataset, batch_size=256, shuffle=False)
            return test_loader, heldout_loader, scalers, global_perf_dict
        else:
            return test_loader, scalers, global_perf_dict
    else:
        if heldout:
            return test_dataset, heldout_dataset, scalers, global_perf_dict
        else:
            return test_dataset, scalers, global_perf_dict


def forward_and_extract(model, batch):
    out = model(batch)
    y_pred = out.detach().cpu().numpy()
    y_true = batch.y_performance.cpu().numpy()
    mask = batch.performance_mask.cpu().numpy()
    return y_pred, y_true, mask


def unnormalize_and_store(y_pred, y_true, mask, scalers, true_all, pred_all):
    for i, scaler in enumerate(scalers):
        valid = (mask[:, i] == 1) & (y_true[:, i] != 0)
        if np.any(valid):
            y_pred[valid, i] = scaler.inverse_transform(y_pred[valid, i].reshape(-1, 1)).squeeze()
            true_all[i].extend(y_true[valid, i])
            pred_all[i].extend(y_pred[valid, i])
    return y_pred, y_true, true_all, pred_all


def unnormalize(y_pred, y_true, mask, scalers):
    y_pred_out = y_pred.view(-1).numpy()
    for i, scaler in enumerate(scalers):
        if (mask[i] == 1) and (y_true[i] != 0):
            y_pred_out[i] = scaler.inverse_transform(y_pred_out[i].reshape(-1, 1)).squeeze()
    return y_pred_out
