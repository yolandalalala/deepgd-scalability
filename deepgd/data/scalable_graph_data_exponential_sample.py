from .graph_data import GraphData

import torch
import networkx as nx
from collections import defaultdict
import random


class ScalableGraphDataExponentialSample(GraphData):

    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    raw_edge_index: torch.Tensor
    raw_edge_attr: torch.Tensor
    edge_weight: torch.Tensor

    def compute_edge_index_edge_spl(self, G, device, landmarks, sampling_factor, **kwargs):
        lms = self.get_maxmin_landmarks(G, landmarks)
        lmsp = self.get_lmsp(G, lms, device)

        landmark_edge_index = self.get_landmark_edge_index(G, lms, device)
        landmark_edge_spl = self.get_landmark_edge_sp(G, lms, lmsp, device)

        sampled_edge_index = self.sample_random_edge_index(lmsp, sampling_factor, device)
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
        return lm_full_index

    @classmethod
    def get_landmark_edge_sp(cls, G, landmarks, lmsp, device):
        # TODO: use pyg.utils.remove_self_loops and pyg.utils.to_undirected
        lms_t = torch.tensor(landmarks, device=device)
        row_idx = torch.arange(lmsp.shape[0], device=device)
        raw_lm_full_sp = lmsp[torch.meshgrid(row_idx, lms_t)].flatten()
        lm_full_sp = raw_lm_full_sp[raw_lm_full_sp != 0]
        return lm_full_sp

    @classmethod
    def sample_random_edge_index(cls, lmsp, sampling_factor, device):
        all_probs = (lmsp * -sampling_factor).exp()
        edge_index_list = []
        for probs in all_probs:
            sampled_src = torch.nonzero(torch.bernoulli(probs)).flatten()
            sampled_dst = torch.nonzero(torch.bernoulli(probs)).flatten()
            edge_index_list.append(torch.stack(torch.meshgrid(sampled_src, sampled_dst)).view(2, -1))
        edge_index = torch.cat(edge_index_list, dim=1)
        return edge_index[:, edge_index[0] != edge_index[1]]

    @classmethod
    # BUG: lmsp is not gauranteed to be in order
    def approximate_shortest_path_from_lmsp(cls, lmsp, edge_index, device):
        lmsp_row_idx = torch.arange(lmsp.shape[0], device=device)
        lmsp_idx_src = torch.meshgrid(lmsp_row_idx, edge_index[0])
        lmsp_idx_dst = torch.meshgrid(lmsp_row_idx, edge_index[1])
        return (lmsp[lmsp_idx_src] + lmsp[lmsp_idx_dst]).min(dim=0).values

