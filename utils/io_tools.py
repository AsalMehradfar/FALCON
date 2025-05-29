import yaml
import os
import json
import torch
import numpy as np
import random


def load_values(load_path):
    """
    Loads circuit-specific values from a YAML file if it exists.

    Args:
        load_path (str): The path to the values file.

    Returns:
        dict: A dictionary containing the values loaded from the YAML file.
              If the file does not exist, an empty dictionary is returned.
    """

    # Construct the file path for the values YAML file
    file_path = f'{load_path}/values.yaml'
    
    # Check if the specified file exists
    if not os.path.exists(file_path):
        return {}  # Return an empty dictionary if the file is missing
    else:
        return load_yaml(file_path, False)  # Load YAML file contents into a dictionary
        

def load_yaml(yaml_path, not_found=True):
    """
    Loads a YAML configuration file and returns its contents as a Python dictionary.

    Args:
        yaml_path (str): The file path to the YAML file.

    Returns:
        dict: A dictionary containing the parsed contents of the YAML file.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
        yaml.YAMLError: If the YAML file contains invalid syntax or cannot be parsed.
        Exception: For any other unexpected errors during the file reading or parsing process.
    """
    # Check if the file exists
    if not os.path.exists(yaml_path) and not_found:
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    try:
        # Open the specified YAML file in read mode
        with open(yaml_path, "r") as file:
            # Use yaml.safe_load to safely parse the YAML file into a Python dictionary
            # This ensures the contents are interpreted correctly, avoiding unsafe operations
            config_yaml = yaml.safe_load(file)
    except yaml.YAMLError as e:
        # Raise an error if the YAML file cannot be parsed
        raise yaml.YAMLError(f"Error parsing YAML file: {yaml_path}\n{e}")
    except Exception as e:
        # Catch any other unexpected errors
        raise Exception(f"An error occurred while loading the YAML file: {yaml_path}\n{e}")

    # Return the parsed YAML content as a dictionary
    return config_yaml


def load_netlist(load_path):
    """
    Loads a netlist file for a specified circuit.

    Args:
        load_path (str): The path to the netlist file.

    Returns:
        list of str: A list of lines from the netlist file.

    Raises:
        FileNotFoundError: If the specified netlist file does not exist.
    """
    file_path = f'{load_path}/netlist'
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Netlist file not found at: {file_path}")

    # Open and read the file
    with open(file_path) as file:
        netlist = file.readlines()

    return netlist


def save_graph(graph_data, node_mapping, edge_attr_dict, save_path):
    """Save graph data in a pretty JSON format for better readability."""
    # Convert tuple keys to strings
    edge_attr_dict_str = {str(k): v for k, v in edge_attr_dict.items()}

    with open(save_path, "w") as f:
        json.dump({
            "x": graph_data.x.tolist(),
            "edge_index": graph_data.edge_index.tolist(),
            "node_mapping": node_mapping,  
            "edge_attr_dict": edge_attr_dict_str  
        }, f, indent=4)  # <---- Pretty-print JSON


def load_performance_names(yaml_path):
    """Loads performance metric names from the YAML file."""

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Data config file not found at: {yaml_path}")
    
    yaml_data = load_yaml(yaml_path, False)
    
    return list(yaml_data.get("Performance", {}).keys())  # Extract performance metric names


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def convert_tuple_to_list(d):
    if isinstance(d, dict):
        return {k: convert_tuple_to_list(v) for k, v in d.items()}
    elif isinstance(d, tuple):
        return list(d)
    else:
        return d
    

def convert_list_to_tuple(d):
    if isinstance(d, dict):
        return {k: convert_list_to_tuple(v) for k, v in d.items()}
    elif isinstance(d, list) and len(d) == 2 and all(isinstance(x, (int, float)) for x in d):
        return tuple(d)
    else:
        return d