import torch
from torch import nn


class Stress(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, node_pos, edge_index, edge_attr):
        start, end = node_pos[edge_index[0]], node_pos[edge_index[1]]
        eu = (start - end).norm(dim=1)
        d = edge_attr[:, 0]
        edge_stress = eu.sub(d).abs().div(d).square().sum()
        return edge_stress
    