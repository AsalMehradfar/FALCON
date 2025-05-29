import torch
import torch.nn as nn
import torch.nn.functional as F

# ELEMENT_TYPES = [
#     "nmos", "pmos", "resistor", "capacitor", "vsource",
#     "port", "inductor", "balun", "isource"
# ]
ELEMENT_TYPES = [
    "nmos_DG", "nmos_GS", "nmos_DS",
    "pmos_DG", "pmos_GS", "pmos_DS",
    "balun_D1", "balun_D2",
    "resistor", "capacitor", "inductor",
    "vsource", "isource", "port"
]

SCALE_FACTORS = {
    'capacitor': {'c': 1e-12},
    'resistor': {'r': 1e3},
    'inductor': {'l': 1e-9},
    'nmos': {'m': 1.0, 'w': 1e-6},
    'pmos': {'m': 1.0, 'w': 1e-6},
    'vsource': {'dc': 1.0, 'mag': 1.0, 'phase': 1.0},
    'isource': {'dc': 1e-3, 'mag': 1e-3},
    'port': {'dbm': 1.0, 'dc': 1.0, 'freq': 1e9, 'num': 1.0},
    'balun': {'rout': 1.0}
}

element_to_idx = {t: i for i, t in enumerate(ELEMENT_TYPES)}

class EdgeEncoder(nn.Module):
    def __init__(self, out_dim, param_templates, str_params_templates):
        super().__init__()

        # REGION_TYPES = list(str_params_templates['region'].keys())
        SOURCE_TYPES = list(str_params_templates['source_type'].keys())
        # self.region_to_idx = {r: i for i, r in enumerate(REGION_TYPES)}
        self.source_to_idx = {s: i for i, s in enumerate(SOURCE_TYPES)}

        self.type_embed = nn.Embedding(len(ELEMENT_TYPES), out_dim)
        # self.region_embed = nn.Embedding(len(REGION_TYPES), out_dim)
        self.source_embed = nn.Embedding(len(SOURCE_TYPES), out_dim)

        # Create one MLP per element type
        self.param_mlps = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(len(param_templates[t]), out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            ) for t in param_templates
        })

        self.final = nn.Sequential(
            nn.Linear(3 * out_dim, out_dim),
            nn.ReLU(),
        )

        self.param_templates = param_templates

    def forward(self, edge_features):
        type_ids = torch.tensor([element_to_idx.get(e['type'], 0) for e in edge_features], device='cpu')
        # region_ids = torch.tensor([self.region_to_idx.get(e['region'], 0) if e['region'] else 0 for e in edge_features], device='cpu')
        source_ids = torch.tensor([self.source_to_idx.get(e['source_type'], 0) if e['source_type'] else 0 for e in edge_features], device='cpu')

        type_emb = self.type_embed(type_ids)
        # region_emb = self.region_embed(region_ids)
        source_emb = self.source_embed(source_ids)

        param_vecs = []
        for e in edge_features:
            # raise ValueError(e)
            # t = e['type']
            comp_type = e['type']           # e.g., 'nmos_DG'
            base_type = comp_type.split("_")[0]
            params = e['params']  # already a tensor

            param_names = self.param_templates[base_type]
            scale_dict = SCALE_FACTORS.get(base_type, {})

            # Scale each parameter individually
            scaled_values = [
                params[i] / scale_dict.get(param_names[i], 1.0)
                for i in range(len(param_names))
            ]
            # scaled_tensor = torch.tensor(scaled_values, dtype=torch.float32).to(params.device)
            scaled_tensor = torch.stack(scaled_values)

            mlp = self.param_mlps[base_type]
            param_vecs.append(mlp(scaled_tensor.unsqueeze(0)))

        param_vecs = torch.cat(param_vecs, dim=0)  # [num_edges, out_dim]
        concat = torch.cat([type_emb, source_emb, param_vecs], dim=-1)

        # if len(edge_features) > 0:
        #     # Debug print for the first edge
        #     e = edge_features[0]
        #     t = e['type']
        #     region = e.get('region', 'none')
        #     source = e.get('source_type', 'none')
        #     params = e['params']

        #     base_type = t.split("_")[0]
        #     print("\n🔍 [Edge Debug]")
        #     print(f"\nType:         {t}")
        #     print(f"\nRegion:       {region}")
        #     print(f"\nSource Type:  {source}")
        #     print(f"\nBase Type:    {base_type}")
        #     print(f"\nParams:       {params.tolist()}")
        #     print(f"\nParam MLP:    {self.param_mlps[base_type](params.unsqueeze(0)).detach().cpu().numpy().flatten().tolist()}")
        #     print(f"\nType Embed:   {type_emb[0]}")
        #     print(f"\nRegion Embed: {region_emb[0]}")
        #     print(f"\nSource Embed: {source_emb[0]}")

        return self.final(concat)
