import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

from utils.encoders import EdgeEncoder


class CircuitGNN(MessagePassing):
    def __init__(self, hidden_dim, out_dim, param_templates, str_params_templates, num_layers=4):
        super(CircuitGNN, self).__init__(aggr='add')

        self.edge_encoder = EdgeEncoder(hidden_dim, param_templates, str_params_templates)
        self.node_encoder = nn.Linear(in_features=1, out_features=hidden_dim)

        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(
                nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim)
                ])
            )

        # self.edge_weight_mlp = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 2, 1),
        #     nn.Sigmoid()  # optional, to bound weights between 0 and 1
        # )

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        nn.init.normal_(self.output_mlp[3].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_mlp[3].bias)

    def forward(self, data):
        x, edge_index, edge_features, batch = data.x, data.edge_index, data.edge_features, data.batch

        # edge_attr = self.edge_encoder(edge_features)  # [num_edges, hidden_dim]
        if isinstance(edge_features, list) and isinstance(edge_features[0], list):
            edge_features = [ef for sublist in edge_features for ef in sublist]

        edge_attr = self.edge_encoder(edge_features)
        x = self.node_encoder(x)  # [num_nodes, hidden_dim]

        for lin_msg, lin_upd in self.gnn_layers:
            m = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            x = F.relu(lin_upd(m) + x)

        graph_repr = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        return self.output_mlp(graph_repr)

    def message(self, x_j, edge_attr):
        # weight = self.edge_weight_mlp(edge_attr)  # outputs [num_edges, 1]
        # return weight * x_j + edge_attr
        return x_j + edge_attr
