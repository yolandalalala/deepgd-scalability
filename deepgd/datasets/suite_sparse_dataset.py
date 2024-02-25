from .graph_drawing_dataset import GraphDrawingDataset
from ..data import GraphData

import os
import shutil
from functools import cached_property
from typing import Callable, Optional, TypeVar, Iterator, Iterable
from typing_extensions import Unpack

import numpy as np
import ssgetpy
import scipy
import torch
import torch_geometric as pyg
from torch_geometric.data import Dataset
import networkx as nx
from tqdm.auto import tqdm

T = TypeVar("T", bound=GraphData)


class SuiteSparseDataset(GraphDrawingDataset.from_cls(pyg.data.Dataset)):

    min_nodes: int
    max_nodes: int
    limit: int

    def __init__(self, *,
                 min_nodes: int = 300,
                 max_nodes: int = 3000,
                 limit: int = 10000,
                 **kwargs: Unpack[GraphDrawingDataset.Args]):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.limit = limit
        super().__init__(**kwargs)

    @cached_property
    def graph_list(self):
        graphs = []
        for graph in ssgetpy.search(colbounds=(self.min_nodes, self.max_nodes), limit=self.limit):
            if graph.psym > 0:
                graphs.append(graph)
        return graphs

    @property
    def index_file_name(self):
        min_nodes = self.min_nodes
        max_nodes = self.max_nodes
        limit = self.limit
        return f"index({min_nodes=},{max_nodes=},{limit=}).txt"

    @property
    def raw_file_names(self):
        # This prevents downloading the raw files if processed files exist
        if pyg.data.dataset.files_exist(self.processed_paths):
            return []
        return list(map(lambda graph: f"{graph.name.lower()}.mtx", self.graph_list))

    @property
    def data_file_names(self):
        return list(map(lambda name: f"{name}.pt", self.index))

    def get_data_file_name(self, G):
        return f"{G.graph['name']}.pt"

    def download(self):
        for raw_name, raw_path, graph in zip(self.raw_file_names, self.raw_paths, self.graph_list):
            if not os.path.exists(raw_path):
                graph.download(destpath=self.raw_dir, extract=True)
                os.rename(os.path.join(self.raw_dir, graph.name.lower(), raw_name), raw_path)
                shutil.rmtree(os.path.join(self.raw_dir, graph.name.lower()))

    def generate(self) -> Iterable[tuple[str, nx.Graph]]:
        # TODO: skipped generating G if already exists, but need to have a way to write the name to index file
        for raw_path in tqdm(self.raw_paths, desc=f"Load graphs"):
            name = os.path.splitext(os.path.basename(raw_path))[0]
            mat = scipy.io.mmread(raw_path)
            G = nx.from_scipy_sparse_array(mat)
            yield name, G

    def process(self):
        super().process()

    def save_data(self, data_iterable: Iterable[T]) -> None:
        for data in data_iterable:
            print('Saving', data.G.graph['name'])
            torch.save(data, self.get_data_path(data.G))

    def load_data(self):
        pass

    def get(self, idx):
        return torch.load(os.path.join(self.data_dir, f"{self.index[idx]}.pt"))
