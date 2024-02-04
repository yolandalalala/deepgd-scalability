from . import DATA_ROOT
from ..data import GraphData

import os
import shutil
from functools import cached_property
from typing import Callable, Optional, TypeVar, Iterator

import numpy as np
import ssgetpy
import scipy
import torch
import torch_geometric as pyg
from torch_geometric.data import Dataset
import networkx as nx
from tqdm.auto import tqdm


T = TypeVar("T", bound=GraphData)


class SuiteSparseDataset(Dataset):

    DEFAULT_NAME = "SuiteSparse"

    dataset_name: str

    def __init__(self, *,
                 root: str = DATA_ROOT,
                 name: str = DEFAULT_NAME,
                 datatype: type[T],
                 datatype_args: dict = None,
                 min_nodes=300,
                 max_nodes=3000,
                 limit=10000):
        self.dataset_name = name
        self.datatype = datatype
        self.datatype_args = datatype_args or {}
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.limit = limit
        self.index = []
        super().__init__(
            root=os.path.join(root, name),
            transform=self.transform,
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter
        )
        with open(self.processed_paths[0], "r") as index_file:
            self.index = index_file.read().strip().split("\n")

    @cached_property
    def graph_list(self):
        graphs = []
        for graph in ssgetpy.search(colbounds=(self.min_nodes, self.max_nodes), limit=self.limit):
            if graph.psym > 0:
                graphs.append(graph)
        return graphs

    @property
    def index_file_name(self):
        return f"index_{self.min_nodes=}_{self.max_nodes=}_{self.limit=}.txt"

    @property
    def raw_file_names(self):
        return list(map(lambda graph: f"{graph.name}.mtx", self.graph_list))

    @property
    def datatype_name(self):
        # sort the keys to ensure consistent naming
        sorted_keys = sorted(self.datatype_args.keys())
        return self.datatype.__name__ + "(" + ",".join(f"{k}={self.datatype_args[k]}" for k in sorted_keys) + ")"

    @property
    def processed_file_names(self):
        file_names = [self.index_file_name]
        index_path = os.path.join(self.processed_dir, file_names[0])
        if os.path.exists(index_path):
            with open(index_path, "r") as index_file:
                name_list = index_file.read().strip().split("\n")
            file_names.extend(list(map(lambda name: os.path.join(self.datatype_name, f"{name}.pt"), name_list)))
        return file_names

    def download(self):
        for graph in self.graph_list:
            raw_path = os.path.join(self.raw_dir, f"{graph.name}.mtx")
            if not os.path.exists(raw_path):
                graph.download(destpath=self.raw_dir, extract=True)
                os.rename(os.path.join(self.raw_dir, graph.name, f"{graph.name}.mtx"), raw_path)
                shutil.rmtree(os.path.join(self.raw_dir, graph.name))

    def generate(self) -> Iterator[nx.Graph]:
        # TODO: skipped generating G if already exists, but need to have a way to write the name to index file
        for raw_path in tqdm(self.raw_paths, desc=f"Load graphs"):
            name = os.path.splitext(os.path.basename(raw_path))[0]
            mat = scipy.io.mmread(raw_path)
            G = nx.from_scipy_sparse_array(mat).to_undirected()
            G.graph.update(dict(
                name=name,
                dataset=self.dataset_name
            ))
            yield G

    def pre_filter(self, G):
        return nx.is_connected(G)

    def pre_transform(self, G) -> T:
        nx.set_edge_attributes(G, 1, "weight")
        G = nx.convert_node_labels_to_integers(G).to_directed()
        G.remove_edges_from(nx.selfloop_edges(G))
        try:
            return self.datatype(G, **self.datatype_args)
        except Exception as e:
            print(f"Error pre-transforming graph {G.graph['name']}: {e}")
            raise e

    def transform(self, data):
        return data

    def process(self):
        def get_path(G):
            return os.path.join(self.processed_dir, self.datatype_name, f"{G.graph['name']}.pt")

        def filter_cached_and_save_index(G_list):
            name_list = []
            for G in G_list:
                name_list.append(G.graph['name'])
                if not os.path.exists(get_path(G)):
                    yield G
            with open(self.processed_paths[0], "w") as index_file:
                index_file.write("\n".join(name_list))

        G_list = filter(self.pre_filter, self.generate())
        data_list = map(self.pre_transform, filter_cached_and_save_index(G_list))
        if not os.path.exists(os.path.join(self.processed_dir, self.datatype_name)):
            os.makedirs(os.path.join(self.processed_dir, self.datatype_name))
        for data in data_list:
            print('Saving', data.G.graph['name'])
            torch.save(data, get_path(data.G))

    def len(self):
        return len(self.index)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, self.datatype_name, f"{self.index[idx]}.pt"))
