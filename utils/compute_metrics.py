import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error


def compute_classification_metrics(y_true, y_pred, class_names=None):
    """
    Computes major classification metrics.
    Args:
        y_true (array): Ground truth labels
        y_pred (array): Predicted labels
        class_names (list, optional): List of class names
    Returns:
        metrics_df (DataFrame): Table of metrics
    """
    metrics = {}

    # Overall scores
    metrics["Accuracy"] = accuracy_score(y_true, y_pred) * 100
    metrics["Balanced Accuracy"] = balanced_accuracy_score(y_true, y_pred) * 100
    metrics["Macro Precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    metrics["Macro Recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    metrics["Macro F1"] = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    metrics["Micro F1"] = f1_score(y_true, y_pred, average="micro", zero_division=0) * 100

    # Per-class scores
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0) * 100
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0) * 100
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0) * 100

    # Create DataFrame
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]

    per_class_df = pd.DataFrame({
        "Precision (%)": np.round(precision_per_class, 1),
        "Recall (%)": np.round(recall_per_class, 1),
        "F1 Score (%)": np.round(f1_per_class, 1)
    }, index=class_names)

    summary_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Score (%)": [np.round(v, 2) for v in metrics.values()]
    })

    return per_class_df, summary_df


def compute_per_class_accuracy(y_true, y_pred, num_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    class_accuracies = []
    for class_id in range(num_classes):
        mask = y_true == class_id
        if np.sum(mask) == 0:
            acc = 0.0
        else:
            acc = np.mean(y_true[mask] == y_pred[mask])
        class_accuracies.append(acc)
    return class_accuracies


def aggregate_metrics(true_all, pred_all, sample_errors, global_perf_dict, rel_errors_by_metric):
    replacements = {"Hz": "GHz", "W": "mW", "V": "mV"}
    units = {k: replacements.get(v, v) for k, v in global_perf_dict.items()}
    metrics = []
    trimmed_sample_errors = np.sort(sample_errors)[:int(len(sample_errors) * (1-1e-5))]

    for i, name in enumerate(global_perf_dict):
        if len(rel_errors_by_metric[name]) == 0:
            continue
        
        trimmed_rel_err = np.sort(rel_errors_by_metric[name])[:int(len(rel_errors_by_metric[name]) * (1-1e-5))]
        yt, yp = np.array(true_all[i]), np.array(pred_all[i])
        metrics.append({
            "Performance Metric": name,
            "Unit": units[name],
            "R² Score": round(r2_score(yt, yp), 3),
            "MAE": round(compute_mae(yt, yp, global_perf_dict[name]), 3),
            "RMSE": round(compute_rmse(yt, yp, global_perf_dict[name]), 3),
            "Rel Error (%)": round(np.mean(trimmed_rel_err) * 100, 3),
        })

    metrics.append({
        "Performance Metric": "Mean",
        "Unit": "-",
        "R² Score": "-",
        "MAE": "-",
        "RMSE": "-",
        "Rel Error (%)": round(np.mean(trimmed_sample_errors) * 100, 3),
    })
    return metrics


def compute_mae(yt, yp, perf_name):
    mae = mean_absolute_error(yt, yp)
    return unit_convertor(mae, perf_name)


def compute_rmse(yt, yp, perf_name):
    rmse = root_mean_squared_error(yt, yp)
    return unit_convertor(rmse, perf_name)


def update_per_metric_rel_errors(rel_errors_by_metric, rel, valid, global_perf_dict):
    for i, name in enumerate(global_perf_dict.keys()):
        tmp = (valid[:, i])
        rel_errors_by_metric[name].extend(rel[tmp, i])


def compute_relative_errors(y_pred, y_true, mask):
    valid = (mask == 1) & (y_true != 0)
    rel = np.zeros_like(y_true)
    rel[valid] = np.abs(y_pred[valid] - y_true[valid]) / (np.abs(y_true[valid]) + 1e-8)
    sample_errors = np.sum(rel, axis=1) / np.sum(valid, axis=1)
    return rel, valid, sample_errors


def unit_convertor(metric, unit):
    if unit == 'W':
        metric = metric * 1000
    elif unit == 'V':
        metric = metric * 1000
    elif unit == 'Hz':
        metric = metric / 1e9

    return metric