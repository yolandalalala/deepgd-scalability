from ..data import GraphData

from itertools import starmap

import os
from abc import abstractmethod
from typing import TypeVar, Iterable, Optional, TypedDict, Generic
from typing_extensions import Self
import networkx as nx
import torch
import torch_geometric as pyg

DATA_ROOT = "datasets"

# TODO: migrate to py3.12
S = TypeVar("S", bound=pyg.data.Dataset)
T = TypeVar("T", bound=GraphData)


class GraphDrawingDataset(Generic[S, T]):

    class Args(TypedDict):
        root: Optional[str]
        name: Optional[str]
        datatype: type[T]
        datatype_args: Optional[dict]

    @classmethod
    def from_cls(cls, parent_cls: type[S]) -> type[S]:  # better return type hint?

        class _Dataset(cls, parent_cls):

            # TODO: support custom index file
            def __init__(self, *,
                         datatype: type[T],
                         root: Optional[str] = None,
                         name: Optional[str] = None,
                         datatype_args: Optional[dict] = None):
                self.__root = root or DATA_ROOT
                self.__name = name or self.__class__.__name__
                self.__datatype = datatype
                self.__datatype_args = datatype_args or {}
                self.__index = []
                super().__init__(
                    root=os.path.join(self.__root, self.__name),
                    transform=self.transform,
                    pre_transform=self.pre_transform,
                    pre_filter=self.pre_filter
                )
                self.reload_index()
                self.load_data()

            def reload_index(self):
                if os.path.exists(self.index_path):
                    with open(self.index_path, "r") as index_file:
                        self.__index = index_file.read().strip().split("\n")

            # TODO: save index only if it does not exist
            def save_index(self, G_iter: Iterable[nx.Graph]) -> Iterable[nx.Graph]:
                self.__index.clear()
                for G in G_iter:
                    self.__index.append(G.graph['name'])
                    yield G
                with open(self.index_path, "w") as index_file:
                    index_file.write("\n".join(self.__index))

            @property
            def index(self):
                return self.__index

            @property
            @abstractmethod
            def index_file_name(self):
                raise NotImplementedError

            @property
            def index_path(self):
                return os.path.join(self.processed_dir, self.index_file_name)

            @property
            def dataset_name(self) -> str:
                return self.__name

            @property
            def datatype_name(self) -> str:
                # sort the keys to ensure consistent naming
                sorted_keys = sorted(self.__datatype_args.keys())
                return self.__datatype.__name__ + "(" + ",".join(
                    f"{k}={self.__datatype_args[k]}" for k in sorted_keys
                ) + ")"

            @property
            def processed_file_names(self):
                self.reload_index()
                return [self.index_file_name] + list(map(
                    lambda name: os.path.join(self.datatype_name, name),
                    self.data_file_names
                ))

            @abstractmethod
            def get_data_file_name(self, G):
                raise NotImplementedError

            def get_data_path(self, G):
                if file_name := self.get_data_file_name(G):
                    return os.path.join(self.data_dir, file_name)
                return None

            @property
            @abstractmethod
            def data_file_names(self):
                raise NotImplementedError

            @property
            def data_dir(self):
                return os.path.join(self.processed_dir, self.datatype_name)

            @property
            def data_paths(self):
                return list(map(
                    lambda name: os.path.join(self.data_dir, name),
                    self.data_file_names
                ))

            @abstractmethod
            def generate(self) -> Iterable[tuple[str, nx.Graph]]:
                raise NotImplementedError

            def pre_process(self, name: str, G: nx.Graph) -> nx.Graph:
                G = G.to_undirected()
                G.graph.update(dict(
                    name=name,
                    dataset=self.dataset_name
                ))
                return G

            def pre_filter(self, G: nx.Graph) -> bool:
                return nx.is_connected(G)

            def pre_transform(self, G: nx.Graph) -> T:
                nx.set_edge_attributes(G, 1, "weight")
                G = nx.convert_node_labels_to_integers(G).to_directed()
                G.remove_edges_from(nx.selfloop_edges(G))
                try:
                    return self.__datatype(G, **self.__datatype_args)
                except Exception as e:
                    print(f"Error pre-transforming graph {G.graph['name']}: {e}")
                    raise e

            def transform(self, data: T) -> T:
                name = data.G.graph['name']
                gt_pos_file = f'{self.processed_dir}/features/gt_pos/{name}.pt'
                if os.path.exists(gt_pos_file):
                    data.gt_pos = torch.load(gt_pos_file)
                else:
                    layout = nx.drawing.nx_agraph.graphviz_layout(data.G, prog='neato')
                    pos_list = [layout[n] for n in data.G.nodes]
                    data.gt_pos = torch.tensor(pos_list)
                    torch.save(data.gt_pos, gt_pos_file)

                return data

            @abstractmethod
            def save_data(self, data_iterable: Iterable[T]) -> None:
                raise NotImplementedError

            @abstractmethod
            def load_data(self):
                raise NotImplementedError

            def process(self):
                def not_cached(G):
                    if self.get_data_path(G):
                        return not os.path.exists(self.get_data_path(G))
                    return True

                raw_G_iterable = starmap(self.pre_process, self.generate())
                filtered_G_iterable = self.save_index(filter(self.pre_filter, raw_G_iterable))
                G_iterable = filter(not_cached, filtered_G_iterable)
                data_iterable = map(self.pre_transform, G_iterable)
                if not os.path.exists(self.data_dir):
                    os.makedirs(self.data_dir, exist_ok=True)
                self.save_data(data_iterable)

            def len(self):
                return len(self.__index)

        return _Dataset
