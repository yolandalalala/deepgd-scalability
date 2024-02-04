import torch
import networkx as nx
from abc import ABC, abstractmethod

import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Data


class GraphData(Data):

    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    raw_edge_index: torch.Tensor
    raw_edge_attr: torch.Tensor
    edge_weight: torch.Tensor

    def __init__(self, G=None, device='cpu', **kwargs):
        if G is None:
            super().__init__()
            return

        # convert n-d list to np array first to speed up
        init_pos = torch.tensor(np.array(list(nx.drawing.random_layout(G).values())))
        raw_edge_index = pyg.utils.to_undirected(torch.tensor(list(G.edges), device=device).T)
        raw_edge_attr = torch.ones(raw_edge_index.shape[1], 2).to(device)

        super().__init__(
            G=G,
            n=G.number_of_nodes(),
            m=G.number_of_edges(),
            x=init_pos.cpu(),
            raw_edge_index=raw_edge_index.cpu(),
            raw_edge_attr=raw_edge_attr.cpu(),
        )

        edge_index, edge_spl = self.compute_edge_index_edge_spl(G, device, **kwargs)
        edge_index, edge_attr = pyg.utils.coalesce(
            edge_index=edge_index,
            edge_attr=self.convert_edge_spl_to_edge_attr(edge_spl),
            reduce='min'  # TODO: upgrade pyg and use 'any'
        )

        self.edge_index = edge_index.cpu()
        self.edge_attr = edge_attr.cpu()
        self.edge_weight = torch.ones_like(edge_attr[:, 0]).cpu()

        self.full_edge_index = self.edge_index
        self.full_edge_attr = self.edge_attr

    @abstractmethod
    def compute_edge_index_edge_spl(self, G, device, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def convert_edge_spl_to_edge_attr(cls, edge_spl):
        return torch.stack([edge_spl, 1 / (edge_spl ** 2)], dim=1)
