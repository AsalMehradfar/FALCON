import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import pandas as pd
import numpy as np

from utils.io_tools import load_yaml
from utils.visual_utils import plot_relative_error_distribution_with_stats

# Global list of all 16 standard performance metrics
ALL_PERF_KEYS = list(load_yaml("./config/data_config.yaml")["Performance"].keys())

def compute_relative_error(pred, true):
    if true == 0:
        return 0.0
    return abs(pred - true) / abs(true)


def compute_nonzero_mean(series_of_lists):
    means = []
    for lst in series_of_lists:
        arr = np.array(lst)
        nonzero_vals = arr[arr != 0]
        if len(nonzero_vals) > 0:
            mean_val = nonzero_vals.mean()
        else:
            mean_val = 0.0
        means.append(mean_val)
    return means


def convert_df_to_16d(df):
    # Step 1: Detect valid performance metrics from column names
    perf_org_cols = [col for col in df.columns if col.startswith('perf_org_')]
    valid_keys = [col.replace('perf_org_', '') for col in perf_org_cols]

    org_vecs = []
    pred_vecs = []
    sim_vecs = []
    rel_errors = []
    idxs = []

    for idx, row in df.iterrows():
        org_vec = np.zeros(16)
        pred_vec = np.zeros(16)
        sim_vec = np.zeros(16)
        rel_vec_sim = np.zeros(16)
        rel_pred = row.get('rel_error_pred')
        if rel_pred < 10:
            idxs.append(idx)

        for i, key in enumerate(ALL_PERF_KEYS):
            if key in valid_keys:
                org_val = row.get(f'perf_org_{key}', 0.0)
                pred_val = row.get(f'perf_pred_{key}', 0.0)
                sim_val = row.get(f'perf_sim_{key}', 0.0)

                org_vec[i] = org_val
                pred_vec[i] = pred_val
                sim_vec[i] = sim_val
                rel_vec_sim[i] = compute_relative_error(sim_val, org_val)

        org_vecs.append(org_vec) 
        pred_vecs.append(pred_vec)
        sim_vecs.append(sim_vec)
        rel_errors.append(rel_vec_sim)

        
    return {
        'perf_org_16d': np.stack(org_vecs),
        'perf_pred_16d': np.stack(pred_vecs),
        'perf_sim_16d': np.stack(sim_vecs),
        'rel_error_pred_16d': rel_errors
    }, idxs


CLASSES = load_yaml("./config/data_config.yaml")["Classes"]
rel_errs = []
all_good = []
rel_dict = {}
for k, v in CLASSES.items():
    try:
        df = pd.read_csv(f"./results/sim/{v}_{k}_simulated.csv")
        results, idxs = convert_df_to_16d(df)
        df['rel_error_pred_16d'] = list(results['rel_error_pred_16d'])
        df['rel_error_mean'] = compute_nonzero_mean(df['rel_error_pred_16d'])
        rel_errs.extend(list(df['rel_error_mean'][idxs]))
    except:
        continue

plot_relative_error_distribution_with_stats(rel_errs, save_path="./results/aggregated.pdf")
print(np.mean(rel_errs))
print(len(rel_errs))
success = len([r for r in rel_errs if r < 0.2])
print(round(success/len(rel_errs)*100, 2))

# df = pd.read_csv(f"./results/sim/16_IFVCO_simulated.csv")
# results, idxs = convert_df_to_16d(df)
# df['rel_error_pred_16d'] = list(results['rel_error_pred_16d'])
# df['rel_error_mean'] = compute_nonzero_mean(df['rel_error_pred_16d'])
# tmp = list(df['rel_error_mean'])
# print(tmp.index(min(tmp)))
# print(df.iloc[186])