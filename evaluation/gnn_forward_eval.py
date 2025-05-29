import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import torch
import pandas as pd
from tqdm import tqdm

from utils.io_tools import seed_everything
from utils.visual_utils import plot_relative_error_distribution, plot_relative_error_distribution_with_stats
from utils.model_utils import load_data, load_model, forward_and_extract, unnormalize_and_store
from utils.compute_metrics import aggregate_metrics, update_per_metric_rel_errors, compute_relative_errors


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, val_loader, scalers, global_perf_dict):
    num_metrics = len(global_perf_dict)
    true_all = [[] for _ in range(num_metrics)]
    pred_all = [[] for _ in range(num_metrics)]
    sample_errors = []
    rel_errors_by_metric = {name: [] for name in global_perf_dict}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating Test", leave=False):
            y_pred, y_true, mask = forward_and_extract(model, batch)
            y_pred, y_true, true_all, pred_all = unnormalize_and_store(
                y_pred, y_true, mask, scalers, true_all, pred_all
            )

            batch_rel, batch_valid, batch_sample_errors = compute_relative_errors(y_pred, y_true, mask)
            sample_errors.extend(batch_sample_errors)

            update_per_metric_rel_errors(rel_errors_by_metric, batch_rel, batch_valid, global_perf_dict)

    metrics = aggregate_metrics(true_all, pred_all, sample_errors, global_perf_dict, rel_errors_by_metric)
    return pd.DataFrame(metrics), sample_errors, rel_errors_by_metric


def main():

    test_loader, heldout_loader, scalers, global_perf_list = load_data()
    model = load_model(device)

    df, errs, rel_errors_by_metric = evaluate(model, test_loader, scalers, global_perf_list)
    plot_relative_error_distribution_with_stats(errs, save_path=f'./plots/test_rel_err/test_rel_err.pdf')
    for k in list(rel_errors_by_metric.keys()):
        plot_relative_error_distribution_with_stats(rel_errors_by_metric[k], title=f'{k}', save_path=f'./plots/test_rel_err/{k}.pdf')
    
    df_heldout, errs_heldout, heldout_rel_errors_by_metric = evaluate(model, heldout_loader, scalers, global_perf_list)
    plot_relative_error_distribution_with_stats(errs_heldout, save_path="./plots/heldout_rel_err/heldout_rel_err.pdf")
    for k in list(heldout_rel_errors_by_metric.keys()):
        if not heldout_rel_errors_by_metric[k]:
            continue
        plot_relative_error_distribution_with_stats(heldout_rel_errors_by_metric[k], title=f'{k}', save_path=f'./plots/heldout_rel_err/{k}.pdf')

    # Display and save
    print("\n📊 GNN Evaluation on Test Set")
    print(df.to_markdown(index=False))
    df.to_csv("./results/gnn_test_metrics.csv", index=False)

    # Display and save
    print("\n📊 GNN Evaluation on Heldout Set")
    print(df_heldout.to_markdown(index=False))
    df_heldout.to_csv("./results/gnn_heldout_metrics.csv", index=False)

if __name__ == "__main__":
    main()
