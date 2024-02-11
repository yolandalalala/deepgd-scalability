from torch_geometric.data.data import BaseData

from .graph_drawing_dataset import GraphDrawingDataset
from ..data import GraphData

import os
import re
import shutil
from functools import cached_property
from typing import Callable, Optional, TypeVar, Iterator, Type, Iterable
from typing_extensions import Unpack

import torch
import torch_geometric as pyg
from torch_geometric.data import Dataset, Data
import networkx as nx
from tqdm.auto import tqdm

T = TypeVar("T", bound=GraphData)


class RomeDataset(GraphDrawingDataset.from_cls(pyg.data.InMemoryDataset)):

    URL = "https://www.graphdrawing.org/download/rome-graphml.tgz"
    GRAPH_NAME_REGEX = re.compile(r"grafo(\d+)\.(\d+)")

    def __init__(self, **kwargs: Unpack[GraphDrawingDataset.Args]):
        super().__init__(**kwargs)

    def _parse_metadata(self, logfile: str) -> Iterator[str]:
        with open(logfile) as fin:
            for line in fin.readlines():
                if match := self.GRAPH_NAME_REGEX.search(line):
                    yield match.group(0)

    @property
    def index_file_name(self):
        return "index.txt"

    @property
    def raw_file_names(self) -> list[str]:
        metadata_file = "rome/Graph.log"
        if os.path.exists(metadata_path := os.path.join(self.raw_dir, metadata_file)):
            return list(map(lambda f: f"rome/{f}.graphml", self._parse_metadata(metadata_path)))
        return [metadata_file]

    @property
    def data_file_names(self):
        return ["data.pt"]

    def get_data_file_name(self, G):
        return None

    def download(self) -> None:
        pyg.data.download_url(self.URL, self.raw_dir)
        pyg.data.extract_tar(f'{self.raw_dir}/rome-graphml.tgz', self.raw_dir)

    def generate(self) -> Iterator[nx.Graph]:
        def key(path):
            match = self.GRAPH_NAME_REGEX.search(path)
            return int(match.group(1)), int(match.group(2))

        for file in tqdm(sorted(self.raw_paths, key=key), desc=f"Loading graphs"):
            name = self.GRAPH_NAME_REGEX.search(file).group(0)
            G = nx.read_graphml(file)
            yield name, G

    def process(self):
        super().process()

    def save_data(self, data_iterable: Iterable[T]) -> None:
        self.save(list(data_iterable), self.data_paths[0])

    def load_data(self):
        self.load(self.data_paths[0])