import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import numpy as np
import torch
import joblib
from tqdm import tqdm

from data_modules.circuit_dataset import CircuitGraphDataset
from utils.data_utils import CircuitGraphWithNormalizedY
from utils.io_tools import load_yaml


if __name__ == "__main__":

    root_dir = "dataset"
    config = load_yaml(os.path.join("config", "data_config.yaml"))
    global_perf_list = list(config["Performance"].keys())
    circuit_to_code = config["Classes"]

    print("[INFO] Loading splits from data_splits.npz...")
    splits = np.load(os.path.join(root_dir, "data_splits.npz"))
    train_idx = list(splits["train"])
    val_idx = list(splits["val"])
    test_idx = list(splits["test"])
    print("[INFO] Loaded split indices from data_splits.npz")

    print("[INFO] Loading base dataset...")
    full_dataset = CircuitGraphDataset(root_dir, circuit_to_code, global_perf_list, edge_attr_indices=test_idx)
    print(f"[INFO] Total samples in dataset: {len(full_dataset)}")

    print(f"[INFO] Before exclusion: {len(train_idx)} train samples, {len(val_idx)} validation samples, {len(test_idx)} test samples.")
    heldout_circuit_name = "RVCO"  # or whatever name you use
    heldout_train_idx = []
    heldout_val_idx = []
    heldout_test_idx = []

    for i in tqdm(range(len(full_dataset)), desc="Splitting", leave=False):
        circuit_type = full_dataset[i].circuit_type

        if circuit_type == heldout_circuit_name:
            if i in train_idx:
                train_idx.remove(i)
                heldout_train_idx.append(i)
            elif i in val_idx:
                val_idx.remove(i)
                heldout_val_idx.append(i)
            else:
                test_idx.remove(i)
                heldout_test_idx.append(i)

    print(f"[INFO] After exclusion: {len(train_idx)} train samples, {len(val_idx)} validation samples, {len(test_idx)} test samples.")

    print("[INFO] Creating + fitting scaler on train set...")
    # Load scaler
    scaler_path = f"./{root_dir}/performance_scaler_gnn.pkl"
    scaler = joblib.load(scaler_path)
    train_dataset = CircuitGraphWithNormalizedY(full_dataset, train_idx, scaler=scaler)

    # scaler = train_dataset.scaler
    # joblib.dump(scaler, os.path.join(root_dir, "performance_scaler_gnn.pkl"))
    # print("[INFO] Saved scaler to performance_scaler_gnn.pkl")

    print("[INFO] Creating validation sets...")
    val_dataset = CircuitGraphWithNormalizedY(full_dataset, val_idx, scaler=scaler)

    torch.save({
        "train": train_dataset,
        "val": val_dataset,
    }, os.path.join(root_dir, "gnn_data.pt"))

    del train_dataset
    del train_idx
    del val_dataset
    del val_idx

    heldout_train_dataset = CircuitGraphWithNormalizedY(full_dataset, heldout_train_idx, scaler=scaler)
    heldout_val_dataset = CircuitGraphWithNormalizedY(full_dataset, heldout_val_idx, scaler=scaler)
    del heldout_train_idx
    del heldout_val_idx

    print("[INFO] Creating test sets...")
    test_dataset = CircuitGraphWithNormalizedY(full_dataset, test_idx, scaler=scaler)
    heldout_test_dataset = CircuitGraphWithNormalizedY(full_dataset, heldout_test_idx, scaler=scaler)

    torch.save({
        "test": test_dataset
    }, os.path.join(root_dir, "gnn_test_data.pt"))

    torch.save({
        "train": heldout_train_dataset,
        "val": heldout_val_dataset,
        "test": heldout_test_dataset
    }, os.path.join(root_dir, "gnn_heldout_data.pt"))

    print("[INFO] Saved preprocessed GNN dataset.")
