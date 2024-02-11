from .graph_data import GraphData

import torch
import networkx as nx
import torch_geometric as pyg


class DenseGraphData(GraphData):

    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    raw_edge_index: torch.Tensor
    raw_edge_attr: torch.Tensor
    edge_weight: torch.Tensor

    def compute_edge_index_edge_spl(self, G, device, **kwargs):
        n = G.number_of_nodes()
        edges = nx.to_directed(nx.complete_graph(n)).edges
        edge_index = torch.tensor(list(edges), device=device).T
        edge_spl = self.compute_apsp(G, device)[tuple(edge_index)]

        return edge_index, edge_spl

    @classmethod
    def compute_apsp(cls, G, device):
        apsp = torch.empty(G.number_of_nodes(), device=device).diag()
        for i, sp in nx.all_pairs_shortest_path_length(G):
            for j, d in sp.items():
                apsp[i, j] = d
        return apsp
