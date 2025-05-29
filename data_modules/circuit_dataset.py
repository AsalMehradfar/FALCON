import os
import torch
import pandas as pd
from torch_geometric.data import Dataset, Data
from collections import defaultdict
import yaml
import json
import ast
import re
import math

from utils.io_tools import load_yaml


CATEGORICAL_KEYS = load_yaml("config/data_config.yaml")["Categorical_Attr"]  # exclude from param templates


def separate_parameters_and_performance(csv_path, performance_names):
    """
    Separates columns into parameters and performance metrics.
    
    Args:
        csv_path (str): Path to the CSV file.
        performance_names (list): List of known performance metric names.
    
    Returns:
        dict: {"parameters": [...], "performance": [...]}
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Get all column names
    all_columns = list(df.columns)

    # Separate columns
    performance_cols = [col for col in all_columns if col in performance_names]
    parameter_cols = [col for col in all_columns if col not in performance_names]

    return {"parameters": parameter_cols, "performance": performance_cols}


def build_param_templates(root_dir):
    param_templates = defaultdict(set)

    for circuit_type in os.listdir(root_dir):
        circuit_path = os.path.join(root_dir, circuit_type)
        if not os.path.isdir(circuit_path):
            continue

        for subfolder in os.listdir(circuit_path):
            subfolder_path = os.path.join(circuit_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            graph_file = os.path.join(subfolder_path, "graph.json")
            if not os.path.exists(graph_file):
                continue

            with open(graph_file, "r") as f:
                graph_data = json.load(f)
                edge_attr_dict = {ast.literal_eval(k): v for k, v in graph_data["edge_attr_dict"].items()}

            for edge_data in edge_attr_dict.values():
                component = edge_data["component"]
                base_type = component.split("_")[0] if "_" in component else component

                keys = list(edge_data.get("numeric_attrs", {}).keys()) + \
                       list(edge_data.get("parametric_attrs", {}).keys())
                filtered_keys = [k for k in keys if k not in CATEGORICAL_KEYS]
                param_templates[base_type].update(filtered_keys)

    with open(os.path.join(root_dir, "param_templates.yaml"), "w") as f:
        yaml.dump({k: sorted(list(v)) for k, v in param_templates.items()}, f)

    return {k: sorted(list(v)) for k, v in param_templates.items()}


# def resolve_edge_features_with_grad(edge_attr_list, x_params, param_names, param_templates):
#     edge_features = []

#     for attr in edge_attr_list:
#         component = attr["component"]
#         base_type = component.split("_")[0] if "_" in component else component
#         source_type = attr.get("numeric_attrs", {}).get("type", "none")

#         param_keys = param_templates.get(base_type, [])
#         param_vector = []

#         for param in param_keys:
#             if "numeric_attrs" in attr and param in attr["numeric_attrs"]:
#                 try:
#                     val = float(attr["numeric_attrs"][param])
#                     # ✅ Keep val in the graph using x_params-based tensor
#                     val_tensor = x_params.new_tensor([val], requires_grad=True)[0]
#                 except:
#                     val_tensor = x_params.new_tensor([0.0], requires_grad=True)[0]

#             elif "parametric_attrs" in attr and param in attr["parametric_attrs"]:
#                 symbol = attr["parametric_attrs"][param]
#                 if symbol in param_names:
#                     idx = param_names.index(symbol)
#                     val_tensor = x_params[idx]  # ✅ connected to autograd
#                 else:
#                     val_tensor = x_params.new_tensor([0.0], requires_grad=True)[0]

#             else:
#                 val_tensor = x_params.new_tensor([0.0], requires_grad=True)[0]

#             param_vector.append(val_tensor)

#         param_tensor = torch.stack(param_vector)

#         edge_features.append({
#             "type": component,
#             "source_type": source_type,
#             "params": param_tensor
#         })

#     return edge_features


def resolve_edge_features_with_grad(edge_attr_list, x_params, param_names, param_templates):
    edge_features = []

    for attr in edge_attr_list:
        component = attr["component"]
        base_type = component.split("_")[0] if "_" in component else component
        source_type = attr.get("numeric_attrs", {}).get("type", "none")

        param_keys = param_templates.get(base_type, [])
        param_vector = []

        for param in param_keys:
            val_tensor = None

            # Priority 1: numeric_attrs
            if "numeric_attrs" in attr and param in attr["numeric_attrs"]:
                try:
                    val = float(attr["numeric_attrs"][param])
                    val_tensor = x_params.new_tensor([val], requires_grad=True)[0]
                except Exception:
                    val_tensor = x_params.new_tensor([0.0], requires_grad=True)[0]

            # Priority 2: parametric_attrs
            elif "parametric_attrs" in attr and param in attr["parametric_attrs"]:
                symbol = attr["parametric_attrs"][param]
                if symbol in param_names:
                    idx = param_names.index(symbol)
                    val_tensor = x_params[idx]
                else:
                    val_tensor = x_params.new_tensor([0.0], requires_grad=True)[0]

            # ✅ Priority 3: computing_attrs (only if param in param_keys)
            elif "computing_attrs" in attr and param in attr["computing_attrs"]:
                expr = attr["computing_attrs"][param]

                # Replace symbolic names with x_params[]
                for symbol in param_names:
                    idx = param_names.index(symbol)
                    expr = re.sub(rf'\b{re.escape(symbol)}\b', f'x_params[{idx}]', expr)

                # Replace units like "600m" → "600e-3"
                expr = re.sub(r'(\d+(?:\.\d+)?)([fpnumkMG]?)', lambda m: str(float(m[1]) * unit_scale(m[2])), expr)
                expr = re.sub(r'x_params\[(\d+)\.0\]', r'x_params[\1]', expr)
                expr = re.sub(r'(?<!\w)(\d+\.?\d*e?-?\d*)(?!\w)', r'torch.tensor(\1)', expr)
                # Evaluate using PyTorch-safe functions
                try:
                    val_tensor = eval(
                        expr,
                        {
                            "x_params": x_params,
                            "sqrt": torch.sqrt,
                            "pow": torch.pow,
                            "exp": torch.exp,
                            "log": torch.log,
                            "torch": torch,  # ✅ Include this!
                        },
                    )
                except Exception as e:
                    print(f"[WARN] Could not evaluate computing_attr for {param}: {expr} → {e}")
                    val_tensor = x_params.new_tensor([0.0], requires_grad=True)[0]

            # Fallback
            else:
                val_tensor = x_params.new_tensor([0.0], requires_grad=True)[0]

            param_vector.append(val_tensor)

        param_tensor = torch.stack(param_vector)
        edge_features.append({
            "type": component,
            "source_type": source_type,
            "params": param_tensor
        })

    return edge_features


def unit_scale(prefix):
    return {
        'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        '': 1.0, 'k': 1e3, 'M': 1e6, 'G': 1e9
    }.get(prefix, 1.0)


def resolve_edge_features(edge_attr_list, x_params, param_names, param_templates):
    edge_features = []

    for attr in edge_attr_list:
        component = attr["component"]
        base_type = component.split("_")[0] if "_" in component else component
        source_type = attr.get("numeric_attrs", {}).get("type", "none")

        param_vector = []
        param_keys = param_templates.get(base_type, [])

        for param in param_keys:
            val = 0.0

            if "numeric_attrs" in attr and param in attr["numeric_attrs"]:
                try:
                    val = float(attr["numeric_attrs"][param])
                except (ValueError, TypeError):
                    val = 0.0

            elif "parametric_attrs" in attr and param in attr["parametric_attrs"]:
                symbol = attr["parametric_attrs"][param]
                if symbol in param_names:
                    val = x_params[param_names.index(symbol)].item()

            elif "computing_attrs" in attr and param in attr["computing_attrs"]:
                expr = attr["computing_attrs"][param]

                # Replace symbols with x_params[index]
                for i, symbol in enumerate(param_names):
                    expr = re.sub(rf'\b{re.escape(symbol)}\b', f'x_params[{i}]', expr)

                # Replace units like 600m → 600e-3
                expr = re.sub(r'(\d+(?:\.\d+)?)([fpnumkMG]?)', lambda m: str(float(m[1]) * unit_scale(m[2])), expr)
                expr = re.sub(r'x_params\[(\d+)\.0\]', r'x_params[\1]', expr)

                try:
                    val = eval(expr, {
                        "x_params": x_params.cpu().numpy(),
                        "sqrt": math.sqrt,
                        "pow": pow,
                        "exp": math.exp,
                        "log": math.log,
                    })
                except Exception as e:
                    print(f"[WARN] Failed to eval computing_attr for {param}: {expr} → {e}")
                    val = 0.0

            param_vector.append(val)

        edge_features.append({
            "type": component,
            "source_type": source_type,
            "params": torch.tensor(param_vector, dtype=torch.float)
        })

    return edge_features


# def resolve_edge_features(edge_attr_list, x_params, param_names, param_templates):
#     edge_features = []

#     # for key, attr in edge_attr_dict.items():
#     for attr in edge_attr_list:
#         component = attr["component"] # 'nmos_DG', 'nmos_GS', etc.
#         base_type = component.split("_")[0] if "_" in component else component
#         source_type = attr.get("numeric_attrs", {}).get("type", "none")

#         param_vector = []
#         param_keys = param_templates.get(base_type, [])

#         for param in param_keys:
#             val = 0.0
#             if "numeric_attrs" in attr and param in attr["numeric_attrs"]:
#                 val_raw = attr["numeric_attrs"][param]
#                 try:
#                     val = float(val_raw)
#                 except (ValueError, TypeError):
#                     val = 0.0
#             elif "parametric_attrs" in attr and param in attr["parametric_attrs"]:
#                 symbol = attr["parametric_attrs"][param]
#                 if symbol in param_names:
#                     val = x_params[param_names.index(symbol)].item()
#             param_vector.append(val)

#         edge_features.append({
#             "type": component,
#             "source_type": source_type,
#             "params": torch.tensor(param_vector, dtype=torch.float)
#         })

#     return edge_features


def build_str_params_templates(root_dir):
    region_vocab = {}
    source_type_vocab = {}
    fundname_vocab = {}

    region_set = set()
    region_set.add('none')
    source_type_set = set()
    source_type_set.add('none')
    fundname_set = set()
    fundname_set.add('none')

    for circuit_type in os.listdir(root_dir):
        circuit_path = os.path.join(root_dir, circuit_type)
        if not os.path.isdir(circuit_path):
            continue

        for subfolder in os.listdir(circuit_path):
            subfolder_path = os.path.join(circuit_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            graph_file = os.path.join(subfolder_path, "graph.json")
            if not os.path.exists(graph_file):
                continue

            with open(graph_file, "r") as f:
                graph_data = json.load(f)
                edge_attr_dict = {ast.literal_eval(k): v for k, v in graph_data["edge_attr_dict"].items()}
        
                for _, attr in edge_attr_dict.items():
                    numeric = attr.get("numeric_attrs", {})
                    if "region" in numeric:
                        region_set.add(numeric["region"])
                    if "type" in numeric:
                        source_type_set.add(numeric["type"])
                    if "fundname" in numeric:
                        fundname_set.add(numeric["fundname"])

    region_vocab = {v: i for i, v in enumerate(sorted(region_set))}
    source_type_vocab = {v: i for i, v in enumerate(sorted(source_type_set))}
    fundname_vocab = {v: i for i, v in enumerate(sorted(fundname_set))}

    with open(os.path.join(root_dir, "str_params_templates.yaml"), "w") as f:
        yaml.dump({'region': region_vocab, 'source_type': source_type_vocab, 'fundname': fundname_vocab}, f)

    return {'region': region_vocab, 'source_type': source_type_vocab, 'fundname': fundname_vocab}


def regenerate_edge_features(data, new_x_params=None, scale_vector=None, param_templates=None):
    
    x_params = new_x_params if new_x_params is not None else data.x_params
    if scale_vector is not None:
        scale_tensor = torch.as_tensor(scale_vector, dtype=x_params.dtype, device=x_params.device)
        x_params = x_params * scale_tensor

    edge_features = resolve_edge_features_with_grad(data.flat_edge_attrs, x_params, data.param_names, param_templates)

    data.x_params = x_params
    data.edge_features = edge_features
    return data


class CircuitGraphDataset(Dataset):
    def __init__(self, root_dir, circuit_to_code, global_performance_list, transform=None, pre_transform=None, edge_attr_indices=None):
        super(CircuitGraphDataset, self).__init__(root_dir, transform, pre_transform)
        self.circuit_to_code = circuit_to_code
        self.global_performance_list = global_performance_list
        self.data_paths = []
        self.edge_attr_indices = set(edge_attr_indices) if edge_attr_indices else set()

        param_template_path = os.path.join(root_dir, "param_templates.yaml")
        if os.path.exists(param_template_path):
            self.param_templates = load_yaml(param_template_path)
        else:
            self.param_templates = build_param_templates(root_dir)

        str_params_template_path = os.path.join(root_dir, "str_params_templates.yaml")
        if os.path.exists(str_params_template_path):
            self.str_params_templates = load_yaml(str_params_template_path)
        else:
            self.str_params_templates = build_str_params_templates(root_dir)

        for circuit_type in os.listdir(root_dir):
            circuit_path = os.path.join(root_dir, circuit_type)
            if not os.path.isdir(circuit_path):
                continue

            for subfolder in os.listdir(circuit_path):
                subfolder_path = os.path.join(circuit_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                graph_file = os.path.join(subfolder_path, "graph.json")
                csv_path = os.path.join(subfolder_path, "dataset.csv")

                if not os.path.exists(graph_file) or not os.path.exists(csv_path):
                    continue

                df = pd.read_csv(csv_path)

                for idx in range(len(df)):
                    self.data_paths.append((graph_file, df.iloc[idx], subfolder))

    def len(self):
        return len(self.data_paths)

    def get(self, idx):
        graph_file, row, circuit = self.data_paths[idx]

        with open(graph_file, "r") as f:
            graph_data = json.load(f)

        node_features = torch.tensor(graph_data["x"], dtype=torch.float)
        node_mapping = graph_data.get("node_mapping", {})

        row_dict = row.to_dict()
        performance_cols = [p for p in row_dict if p in self.global_performance_list]
        parameter_cols = [p for p in row_dict if p not in self.global_performance_list]

        x_params = torch.tensor([row[p] for p in parameter_cols], dtype=torch.float)

        y_performance = torch.tensor(
            [float(row[p]) if p in performance_cols else 0.0 for p in self.global_performance_list],
            dtype=torch.float
        )

        y_raw_performance = torch.tensor([float(row[p]) for p in performance_cols], dtype=torch.float)
        performance_mask = torch.tensor([1.0 if p in performance_cols else 0.0 for p in self.global_performance_list], dtype=torch.float)

        circuit_code = self.circuit_to_code.get(circuit, -1)

        # 1. Load and flatten edge_attr_dict
        edge_attr_dict = {ast.literal_eval(k): v for k, v in graph_data["edge_attr_dict"].items()}

        flat_edge_index = []
        flat_edge_attrs = []

        for (src, dst, _), attr in edge_attr_dict.items():
            flat_edge_index.append([src, dst])
            flat_edge_attrs.append(attr)

        edge_index = torch.tensor(flat_edge_index, dtype=torch.long).t().contiguous()

        # 2. Convert to resolved edge_features
        edge_features = resolve_edge_features(flat_edge_attrs, x_params, parameter_cols, self.param_templates)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            x_params=x_params,
            param_names=parameter_cols,
            y_performance=y_performance.reshape(1,-1),
            # perf_names=performance_cols,
            performance_mask=performance_mask.reshape(1,-1),
            # y_raw_performance=y_raw_performance,
            circuit_type_code=circuit_code,
            circuit_type=circuit
        )

        # Add non-tensor fields as metadata (if needed outside batching)
        data.edge_features = edge_features
        # data.node_mapping=node_mapping
        if idx in self.edge_attr_indices:
            data.flat_edge_attrs = flat_edge_attrs

        # print("📦 Data created:", data)

        return data


