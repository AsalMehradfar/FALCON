import torch
import networkx as nx
from torch_geometric.data import Data


def networkx_to_pyg(nx_graph):
    """
    Converts a NetworkX MultiGraph with all attributes (numeric and string) into a PyTorch Geometric Data object.

    Args:
        nx_graph (networkx.MultiGraph): The NetworkX graph to be converted.

    Returns:
        tuple: (torch_geometric.data.Data, dict, dict)
               - PyG Data object
               - Node mapping (original names → indices)
               - Dictionary of all attributes (numeric & string)
    """
    # Map node names to indices
    node_mapping = {node: i for i, node in enumerate(nx_graph.nodes())}

    # Create node feature tensor (Modify if real features exist)
    x = torch.tensor([[i] for i in range(len(nx_graph.nodes()))], dtype=torch.float)

    edge_index = []
    edge_attr_dict = {}  # Dictionary to store all attributes

    for u, v, key, data in nx_graph.edges(keys=True, data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])  # Convert nodes to indices

        # Store all attributes (numeric and string) together
        edge_attr_dict[(node_mapping[u], node_mapping[v], key)] = data.copy()

    # Convert edge_index to tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)  # No separate edge_attr

    return data, node_mapping, edge_attr_dict  # Return all attributes together


def pyg_to_networkx(pyg_data, node_mapping, edge_attr_dict):
    """
    Converts a PyTorch Geometric Data object back to a NetworkX MultiGraph, 
    restoring both numeric and string attributes.

    Args:
        pyg_data (torch_geometric.data.Data): The PyG Data object.
        node_mapping (dict): Mapping from original NetworkX node names to PyG indices.
        edge_attr_dict (dict): Stored attributes (numeric & string).

    Returns:
        networkx.MultiGraph: The reconstructed NetworkX MultiGraph.
    """
    # Create an empty MultiGraph
    nx_graph = nx.MultiGraph()

    # Reverse the node mapping (PyG index -> original node name)
    reverse_mapping = {v: k for k, v in node_mapping.items()}

    # Add nodes back
    for i in range(pyg_data.x.shape[0]):
        nx_graph.add_node(reverse_mapping[i])

    # Convert edges back
    edge_index = pyg_data.edge_index.t().tolist()  # Convert tensor to list of edges

    # Reconstruct edges with attributes
    for (u, v) in edge_index:
        key = len(nx_graph.get_edge_data(reverse_mapping[u], reverse_mapping[v], default={}))  # Track multi-edges

        # Restore all attributes (numeric & string)
        attr_dict = edge_attr_dict.get((u, v, key), {})
        nx_graph.add_edge(reverse_mapping[u], reverse_mapping[v], **attr_dict)

    return nx_graph

