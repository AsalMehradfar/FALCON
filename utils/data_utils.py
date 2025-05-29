from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def fit_performance_scaler_ignore_zeros(full_dataset, indices, perf_dim=16):
    perf_values = [[] for _ in range(perf_dim)]

    for i in tqdm(indices, desc="Creating Scalers", leave=False):
        perf = full_dataset[i].y_performance.view(-1).numpy()
        for j in range(perf_dim):
            if perf[j] != 0.0:
                perf_values[j].append(perf[j])

    # Fit separate scaler per dimension
    scalers = []
    for j in range(perf_dim):
        values = np.array(perf_values[j]).reshape(-1, 1)
        scaler = StandardScaler()
        scaler.fit(values)
        scalers.append(scaler)

    return scalers


class PerformanceToClassWithPerMetricScaler(Dataset):
    def __init__(self, base_dataset, indices, scaler=None):
        self.indices = indices
        self.scaler = scaler

        if self.scaler is None:
            print("[INFO] Fitting performance scaler...")
            self.scaler = fit_performance_scaler_ignore_zeros(base_dataset, indices)

        self.samples = []
        for idx in self.indices:
            sample = base_dataset[idx]

            y = sample.y_performance.clone().view(-1)  # [16]
            y_norm = y.clone()

            for i, s in enumerate(self.scaler):
                if y[i] != 0.0:
                    y_norm[i] = torch.tensor(
                        s.transform([[y[i].item()]])[0][0], dtype=torch.float32
                    )

            self.samples.append((y_norm, sample.circuit_type_code))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    

class CircuitGraphWithNormalizedY(Dataset):
    def __init__(self, base_dataset, indices, scaler=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.scaler = scaler

        if self.scaler is None:
            self.scaler = fit_performance_scaler_ignore_zeros(base_dataset, indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        base_idx = self.indices[idx]
        data = self.base_dataset[base_idx]

        y = data.y_performance.clone().view(-1)  # shape: [16]
        y_norm = y.clone()

        for i, s in enumerate(self.scaler):
            if y[i] != 0.0:
                y_norm[i] = torch.tensor(
                    s.transform([[y[i].item()]])[0][0], dtype=torch.float32
                )

        data.y_normalized = y_norm.reshape(1, -1)

        return data
