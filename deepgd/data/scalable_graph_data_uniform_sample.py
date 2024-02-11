from .graph_data import GraphData

import torch
import networkx as nx
from collections import defaultdict
import random


class ScalableGraphDataUniformSample(GraphData):

    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    raw_edge_index: torch.Tensor
    raw_edge_attr: torch.Tensor
    edge_weight: torch.Tensor

    def compute_edge_index_edge_spl(self, G, device, landmarks, random_edges, **kwargs):
        lms = self.get_maxmin_landmarks(G, landmarks)
        lmsp = self.get_lmsp(G, lms, device)

        landmark_edge_index = self.get_landmark_edge_index(G, lms, device)
        landmark_edge_spl = self.get_landmark_edge_sp(G, lms, lmsp, device)

        sampled_edge_index = self.sample_random_edge_index(G, random_edges, device)
        sampled_sp = self.approximate_shortest_path_from_lmsp(lmsp, sampled_edge_index, device)

        edge_index = torch.cat([
            self.raw_edge_index.to(device),
            landmark_edge_index,
            sampled_edge_index
        ], dim=1)

        edge_spl = torch.cat([
            torch.ones(self.m, device=device),
            landmark_edge_spl,
            sampled_sp
        ], dim=0)

        return edge_index, edge_spl

    @classmethod
    def get_maxmin_landmarks(cls, G, k):
        landmarks = [random.choice(list(G.nodes))]
        min_dist = defaultdict(lambda: float('inf'))
        for _ in range(min(k, G.number_of_nodes()) - 1):
            for dst, path in nx.shortest_path(G, landmarks[-1]).items():
                min_dist[dst] = min(len(path) - 1, min_dist[dst])
            landmarks.append(max(min_dist, key=min_dist.__getitem__))
        return landmarks

    @classmethod
    def get_lmsp(cls, G, landmarks, device):
        a = torch.tensor(list(G.nodes), device=device)
        lmsp = []
        for lm in landmarks:
            spl = nx.shortest_path_length(G, lm)
            b = torch.tensor(list(spl.keys()), device=device)
            # get rev_idx such that b[rev_idx] == a
            rev_idx = a.argsort()[b.argsort().argsort()].argsort()
            lmsp.append(torch.tensor(list(spl.values()), device=device)[rev_idx])
        return torch.stack(lmsp)

    # @classmethod
    # # BUG: lmsp is not gauranteed to be in order
    # def approximate_apsp_from_lmsp(cls, lmsp):
    #     apsp = (lmsp[:, None, :] + lmsp[:, :, None]).min(dim=0).values
    #     apsp.fill_diagonal_(0)
    #     return apsp

    @classmethod
    def get_landmark_edge_index(cls, G, landmarks, device):
        # TODO: use pyg.utils.remove_self_loops and pyg.utils.to_undirected
        lms_t = torch.tensor(landmarks, device=device)
        raw_lm_full_index = torch.stack(torch.meshgrid(lms_t, lms_t), dim=0).view(2, -1)
        lm_full_index = raw_lm_full_index[:, raw_lm_full_index[0] != raw_lm_full_index[1]]
        nonlandmarks = list(set(G.nodes) - set(landmarks))
        if nonlandmarks:
            nodes_t = torch.tensor(nonlandmarks, device=device)
            raw_index = torch.stack(torch.meshgrid(lms_t, nodes_t), dim=0).view(2, -1)
            return torch.cat([lm_full_index, raw_index, raw_index.flip(dims=[0])], dim=1)
        return lm_full_index

    @classmethod
    def get_landmark_edge_sp(cls, G, landmarks, lmsp, device):
        # TODO: use pyg.utils.remove_self_loops and pyg.utils.to_undirected
        lms_t = torch.tensor(landmarks, device=device)
        row_idx = torch.arange(lmsp.shape[0], device=device)
        raw_lm_full_sp = lmsp[torch.meshgrid(row_idx, lms_t)].flatten()
        lm_full_sp = raw_lm_full_sp[raw_lm_full_sp != 0]
        nonlandmarks = list(set(G.nodes) - set(landmarks))
        if nonlandmarks:
            nodes_t = torch.tensor(nonlandmarks, device=device)
            raw_sp = lmsp[torch.meshgrid(row_idx, nodes_t)].flatten()
            return torch.cat([lm_full_sp, raw_sp, raw_sp])
        return lm_full_sp

    @classmethod
    def sample_random_edge_index(cls, G, d, device):
        n = G.number_of_nodes()
        # Jeong Han Kim and Van H. Vu, Generating random regular graphs, Proceedings of the
        # thirty-fifth ACM symposium on Theory of computing, San Diego, CA, USA, pp 213–222, 2003.
        # http://portal.acm.org/citation.cfm?id=780542.780576
        # This guarantees that the complexity is O(n)
        d = min(d, int(n ** (1/3)))
        d = max(d if d * n % 2 == 0 else d - 1, 1 if n % 2 == 0 else 2)
        random_edges = nx.to_directed(nx.random_regular_graph(d, n)).edges
        return torch.tensor(list(random_edges), device=device).T

    @classmethod
    # BUG: lmsp is not gauranteed to be in order
    def approximate_shortest_path_from_lmsp(cls, lmsp, edge_index, device):
        lmsp_row_idx = torch.arange(lmsp.shape[0], device=device)
        lmsp_idx_src = torch.meshgrid(lmsp_row_idx, edge_index[0])
        lmsp_idx_dst = torch.meshgrid(lmsp_row_idx, edge_index[1])
        return (lmsp[lmsp_idx_src] + lmsp[lmsp_idx_dst]).min(dim=0).values
