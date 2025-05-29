import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import torch
import numpy as np
import os
from copy import copy
import json

from utils.io_tools import load_yaml, seed_everything, convert_list_to_tuple
from utils.visual_utils import plot_relative_error_distribution, plot_loss_backward
from utils.model_utils import load_data, load_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.backward_utils import *


with open("./config/clamp_bounds.json", "r") as f:
    CLAMP_BOUNDS_BY_TOPOLOGY = convert_list_to_tuple(json.load(f))

YELLOW = "\033[93m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def invert_performance_to_params(model, graph_data, idx, param_templates,
                                 results_dict, global_perf_dict, scalers=None,
                                 lr=1e-2, num_steps=500, verbose=False):
    model.eval()
    device = next(model.parameters()).device
    graph_data = copy(graph_data).to(device)

    x_params = initialize_param_vector(CLAMP_BOUNDS_BY_TOPOLOGY, graph_data, device)
    optimizer = torch.optim.Adam([x_params], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=50)

    best_loss = float("inf")
    best_params = None
    best_param_perf = None
    early_stop_counter = 0
    loss_history = []

    # print(f"{YELLOW}\n\n--- {idx+1}: Circuit Topology {graph_data.circuit_type} ---\n", f"{RESET}")
    # print(f"{CYAN}Initial: [" + ", ".join(f"{x:.2e}" for x in x_params.tolist()) + "]", f"{RESET}")
    # print(f"{MAGENTA}Original: [" + ", ".join(f"{x:.2e}" for x in graph_data.x_params.tolist()) + "]", f"{RESET}")
    # print("Param Names:", graph_data.param_names)

    for step in range(num_steps):
        optimizer.zero_grad()
        loss, out, _ = run_optimization_step(model, graph_data, x_params, param_templates)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        scheduler.step(loss.item())

        clamp_params(CLAMP_BOUNDS_BY_TOPOLOGY, x_params, graph_data)
        log_step_info(step, loss, x_params, verbose, num_steps)

        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            best_params = x_params.clone().detach()
            best_param_perf = out.clone().detach()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= 200:
            # print(f"{RED}[Early Stop] No improvement for {patience} steps. Stopping...", f"{RESET}")
            break

    best_perf_pred, rel_error, area, success = log_final_info(
        graph_data, best_params, best_param_perf, best_loss, scalers, idx, results_dict, global_perf_dict
    )

    # plot_loss_backward(loss_history, save_path=f'./plots/backward_loss/{idx}.pdf', log_scale=False)

    return best_params.detach(), best_perf_pred, best_loss, rel_error, success


def main():

    relative_errors = []
    best_losses = []
    results_dict = {}
    success_dict = {}
    all_dict = {}
    finished = []

    test_dataset, scalers, global_perf_dict = load_data(loader=False, heldout=False)
    model = load_model(device)

    for i, sample in enumerate(test_dataset):
        # Step 1: Get data
        # if i >= 5: ########### For test
        #     break
        sample = copy(sample)

        if sample.circuit_type in finished:
            continue

        # Step 2: Run inversion
        optimized_params, pred_perf, best_loss, rel_error, success = invert_performance_to_params(
            model=model,
            graph_data=sample,
            idx=i,
            global_perf_dict=global_perf_dict,
            results_dict=results_dict,
            scalers=scalers,
            param_templates=load_yaml('./dataset/param_templates.yaml'),
            lr=1e-6,
            num_steps=1000,
            verbose=False
        )

        # Step 3: Compute relative error
        # metrics, rel_error = compute_sample_metrics_relative_err(pred_perf, sample, global_perf_dict)
        # print(metrics)

        if success:
            success_dict[sample.circuit_type] = success_dict.get(sample.circuit_type, 0) + 1
            all_dict[sample.circuit_type] = all_dict.get(sample.circuit_type, 0) + 1
            best_losses.append(best_loss)
            relative_errors.append(rel_error)

            df = results_dict.get(sample.circuit_type)
            if df is not None and len(df) == 500 and sample.circuit_type not in finished:
                finished.append(sample.circuit_type)
                df.to_csv(f"./results/circuits/{topologies[sample.circuit_type]}_{sample.circuit_type}.csv", index=False)
        else:
            success_dict[sample.circuit_type] = success_dict.get(sample.circuit_type, 0)
            all_dict[sample.circuit_type] = all_dict.get(sample.circuit_type, 0) + 1
        # print("Relative Error:", rel_error)
        # print("Final Loss History:", loss)

    # relative_errors = [x / 100 for x in relative_errors]
    # plot_relative_error_distribution(relative_errors, save_path="./plots/backward_pred_rel_err.pdf")
    # for circuit_type, df in results_dict.items():
    #     if circuit_type not in finished:
    #         df.to_csv(f"./results/circuits/{topologies[circuit_type]}_{circuit_type}.csv", index=False)

    # ratios = {}
    # for key, total in all_dict.items():
    #     success = success_dict.get(key, 0)
    #     ratios[key] = success / total if total > 0 else 0.0

    # print(len(best_losses))
    # print(ratios)

if __name__ == "__main__":
    main()