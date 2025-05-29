import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from sklearn.model_selection import train_test_split
import numpy as np

from data_modules.circuit_dataset import CircuitGraphDataset
from utils.io_tools import load_yaml, seed_everything


if __name__ == "__main__":
    seed_everything(42)
    print("[INFO] Splitting dataset indices into train, validation, and test...")

    root_dir = "dataset"
    data_config = load_yaml('./config/data_config.yaml')
    global_perf_list = list(data_config['Performance'].keys())
    circuit_to_code = data_config['Classes']

    # Load full dataset
    full_dataset = CircuitGraphDataset(root_dir, circuit_to_code, global_perf_list)
    all_indices = list(range(len(full_dataset)))
    all_labels = [full_dataset[i].circuit_type_code for i in all_indices]

    # First split: train (70%) vs temp (30%)
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.3,
        stratify=all_labels,
        random_state=42
    )

    # Second split: validation (15%) vs test (15%) from temp
    temp_labels = [full_dataset[i].circuit_type_code for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )

    # After splitting
    np.savez(f"./{root_dir}/data_splits.npz", train=train_idx, val=val_idx, test=test_idx)
    print("[INFO] Saved split indices to dataset/data_splits.npz")