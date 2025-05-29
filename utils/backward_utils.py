import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import torch
import random
import json
from copy import copy
import pandas as pd
import numpy as np

from utils.io_tools import load_yaml
from data_modules.circuit_dataset import regenerate_edge_features
from train.loss import masked_mse_loss, compute_total_layout_area, compute_aggregated_loss
from utils.model_utils import load_data, load_model, unnormalize


YELLOW = "\033[93m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"

# CLAMP_BOUNDS = {
#     'capacitor': {'c': (100e-15, 500e-15)},
#     'inductor': {'l': (100e-12, 500e-12)},
#     'isource': {'dc': (3e-3, 10e-3)},
#     'nmos': {'w': (5e-6, 20e-6)},
#     'pmos': {'w': (3e-6, 5e-6)},
#     'resistor': {'r': (500, 2e3)},
#     'vsource': {'dc': (0.8, 1.2)},
# }


# def build_param_scale_vector_from_edges(flat_edge_attrs, param_names, scale_factors, device):
#     """
#     Look through edge attributes to determine the base type of each symbolic parameter.
#     Use SCALE_FACTORS to get an informed initial guess.
#     """
#     symbol_to_type = {}

#     for attr in flat_edge_attrs:
#         base_type = attr["component"].split("_")[0]

#         if "parametric_attrs" in attr:
#             for k, symbol in attr["parametric_attrs"].items():
#                 if symbol in param_names:
#                     symbol_to_type[symbol] = (base_type, k)

#     init_vals = []
#     for name in param_names:
#         if name in symbol_to_type:
#             base_type, param_key = symbol_to_type[name]
#             scale = scale_factors.get(base_type, {}).get(param_key.lower(), 1.0)
#         elif name in ['L1p', 'L1s', 'L3p', 'L3s']: # for DPA
#             scale = 1e-9
#         else:
#             scale = 1.0
#          # Add small noise: e.g., ±5% of scale
#         noise = scale * random.uniform(-0.1, 0.1)
#         val = scale + noise
#         init_vals.append(val)

#     return torch.nn.Parameter(torch.tensor(init_vals, dtype=torch.float32, device=device))


# def generate_scale_dict(clamp_bounds_dict, topology_name):
#     bounds = clamp_bounds_dict.get(topology_name, {})
#     scale_dict = {}

#     for comp_name, params in bounds.items():
#         scale_dict[comp_name] = {}
#         for param, (low, high) in params.items():
#             scale_dict[comp_name][param] = (low + high) / 2

#     # for comp_type, params in bounds.items():
#     #     scale_dict[comp_type] = {}
#     #     for param, (low, high) in params.items():
#     #         scale_dict[comp_type][param] = (low + high) / 2

#     return scale_dict


# def get_clamp_bounds_from_edges(flat_edge_attrs, param_names):
#     symbol_to_clamp = {}

#     for attr in flat_edge_attrs:
#         base_type = attr["component"].split("_")[0]
#         if "parametric_attrs" not in attr:
#             continue

#         for k, symbol in attr["parametric_attrs"].items():
#             if symbol not in param_names:
#                 continue

#             # ✅ Special case for inductors with predefined param names
#             if symbol in ['L1p', 'L1s', 'L3p', 'L3s']:
#                 symbol_to_clamp[symbol] = CLAMP_BOUNDS['inductor']['l']
#             elif (base_type in CLAMP_BOUNDS) and (k.lower() in CLAMP_BOUNDS[base_type]):
#                 symbol_to_clamp[symbol] = CLAMP_BOUNDS[base_type][k.lower()]

#     # Fallback to default clamp if not found
#     return [symbol_to_clamp.get(name, (1e-16, 1e4)) for name in param_names]


def build_param_scale_vector_from_scaling_dict(param_names, scale_factors, device):
    """
    Builds an initial parameter vector using provided scale_factors for symbolic param names.
    Adds ±10% noise to each scale.
    """
    init_vals = []
    for name in param_names:
        scale = scale_factors.get(name, 1.0)
        noise = scale * random.uniform(-0.1, 0.1)
        val = scale + noise
        init_vals.append(val)

    return torch.nn.Parameter(torch.tensor(init_vals, dtype=torch.float32, device=device))


topologies = load_yaml("./config/data_config.yaml")["Classes"]
# Set limits: DLNA and DohPA get 1.5, others get 1.0
area_thresholds = {
    name: 1.5 if name in ["DLNA", "DohPA", "ClassBPA"] else 1.0
    for name in list(topologies.keys())
}

def is_area_successful(circuit_type, area_mm2):
    return area_mm2 <= area_thresholds.get(circuit_type, 1.0)


def get_clamp_bounds(clamp_bounds, param_names, topology_name):
    comp_bounds = clamp_bounds.get(topology_name, {})
    return [comp_bounds.get(name, (1e-16, 1e4)) for name in param_names]


def inject_params(sample, x_params, param_templates):
    sample = copy(sample)
    sample = regenerate_edge_features(sample, new_x_params=x_params, param_templates=param_templates)
    return sample


def generate_scale_dict(clamp_bounds_dict, topology_name):
    bounds = clamp_bounds_dict.get(topology_name, {})
    scale_dict = {}

    for comp_name, (low, high) in bounds.items():
        scale_dict[comp_name] = (low + high) / 2

    return scale_dict


def initialize_param_vector(clamp_bounds_dict, graph_data, device):
    return build_param_scale_vector_from_scaling_dict(
        param_names=graph_data.param_names,
        scale_factors=generate_scale_dict(clamp_bounds_dict, graph_data.circuit_type),
        device=device
    )

def run_optimization_step(model, graph_data, x_params, param_templates):
    batch_updated = inject_params(graph_data, x_params, param_templates)
    out = model(batch_updated)
    performance_loss = masked_mse_loss(out, graph_data.y_normalized, graph_data.performance_mask)
    area_mm2 = compute_total_layout_area(graph_data.flat_edge_attrs, x_params, graph_data.param_names)
    loss = compute_aggregated_loss(performance_loss, area_mm2)
    return loss, out, area_mm2


def clamp_params(clamp_bounds_dict, x_params, graph_data):
    clamp_bounds = get_clamp_bounds(clamp_bounds_dict, graph_data.param_names, graph_data.circuit_type)
    with torch.no_grad():
        for i, (min_val, max_val) in enumerate(clamp_bounds):
            x_params[i].clamp_(min=min_val, max=max_val)
    # with torch.no_grad():
    #     x_params.clamp_(min=1e-16, max=1e4)


def log_step_info(step, loss, x_params, verbose, num_steps):
    if verbose and (step % 50 == 0 or step == num_steps - 1):
        print(f"\n--- Step {step:03d} ---")
        print(f"Loss: {loss.item():.3f}")
        print("x_params:", x_params.detach().cpu().numpy())    


def log_final_info(graph_data, best_params, best_perf_pred, best_loss, scalers, idx, results_dict, global_perf_dict, rel_thresh=20):
    y_perf = graph_data.y_performance.view(-1).cpu()
    perf_mask = graph_data.performance_mask.view(-1).bool().cpu().numpy()
    used_perf_names = [k for k, m in zip(global_perf_dict.keys(), perf_mask) if m]

    best_perf_pred = unnormalize(best_perf_pred, y_perf, perf_mask, scalers)
    rel_error = get_mean_relative_err(best_perf_pred, graph_data)
    area = compute_total_layout_area(graph_data.flat_edge_attrs, best_params, graph_data.param_names)
    circuit_type = graph_data.circuit_type
    success = rel_error < rel_thresh and is_area_successful(circuit_type, area)

    print(f"{CYAN}Area: {area:.4f}{RESET}")
    print(f"{CYAN}Rel Err: {rel_error:.4f}{RESET}")
    print(f"{CYAN}Loss: {best_loss:.4f}{RESET}")

    if success:
        row = {
            'row_idx': idx,
            **{k: v for k, v in zip(graph_data.param_names, best_params.cpu().numpy())},
            **{f'perf_org_{k}': v for k, v in zip(used_perf_names, y_perf.cpu().numpy()[perf_mask])},
            **{f'perf_pred_{k}': v for k, v in zip(used_perf_names, best_perf_pred[perf_mask])},
            'rel_error_pred': rel_error,
            'area': area
        }
        df_new = pd.DataFrame([row])
        if circuit_type in results_dict:
            results_dict[circuit_type] = pd.concat([results_dict[circuit_type], df_new], ignore_index=True)
        else:
            results_dict[circuit_type] = df_new
        print(f"{GREEN}SUCCESS!", f"{RESET}")
    else:
        print(f"{RED}FAILED!", f"{RESET}")
    return best_perf_pred, rel_error, area, success


def get_mean_relative_err(y_pred, sample):
    y_true = sample.y_performance.view(-1).cpu().numpy()
    mask = sample.performance_mask.view(-1).cpu().numpy()

    # Relative error
    valid = (mask == 1) * (y_true != 0)
    rel = np.zeros_like(y_true)
    rel[valid] = np.abs(y_pred[valid] - y_true[valid]) / (np.abs(y_true[valid]) + 1e-8)

    sample_error = np.sum(rel) / np.sum(valid)
    sample_error = float(sample_error.item()) * 100
    return sample_error


def compute_sample_metrics_relative_err(y_pred, sample, global_perf_dict):
    y_true = sample.y_performance.view(-1).cpu().numpy()
    mask = sample.performance_mask.view(-1).cpu().numpy()

    # Relative error
    valid = (mask == 1) * (y_true != 0)
    rel = np.zeros_like(y_true)
    rel[valid] = np.abs(y_pred[valid] - y_true[valid]) / (np.abs(y_true[valid]) + 1e-8)

    sample_error = np.sum(rel) / np.sum(valid)
    sample_error = float(sample_error.item())

    # Aggregate metrics
    metrics = []
    for i, name in enumerate(list(global_perf_dict.keys())):
        if valid[i] == 0:
            continue
        metrics.append({
            "Performance Metric": name,
            "Rel Error (%)": round((rel[i] / valid[i]) * 100, 3),
        })

    # Add average per-sample error
    metrics.append({
        "Performance Metric": "Mean",
        "Rel Error (%)": round(sample_error * 100, 3),
    })

    return pd.DataFrame(metrics), sample_error