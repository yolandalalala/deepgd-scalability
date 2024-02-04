from smartgd.common.data.base_graph_drawing_data import *

import unittest

import networkx as nx


class TestBaseGraphDrawingData(unittest.TestCase):

    G_list:     list[nx.Graph]
    data_list:  list[BaseGraphDrawingData]
    batch:      Batch  # Batch[BaseGraphDrawingData]
    wheel:      BaseGraphDrawingData
    ladder:     BaseGraphDrawingData
    grid:       BaseGraphDrawingData
    lollipop:   BaseGraphDrawingData

    def setUp(self) -> None:
        BaseGraphDrawingData.optional_fields = [
            "edge_pair_metaindex", "face", "gabriel_index", "rng_index"
        ]
        self.G_list = [
            nx.wheel_graph(10),
            nx.ladder_graph(10),
            nx.grid_graph((4, 4)),
            nx.lollipop_graph(6, 6)
        ]
        for G in self.G_list:
            G.graph.update(dict(
                name="name",
                dataset="dataset_name"
            ))
        self.data_list = list(map(BaseGraphDrawingData, self.G_list))
        (self.wheel,
         self.ladder,
         self.grid,
         self.lollipop) = self.data_list

    @property
    def batch(self) -> Batch:
        return Batch.from_data_list(self.data_list)

    def test_fields(self):
        all_fields = BaseGraphDrawingData.fields()
        pre_transform_fields = BaseGraphDrawingData.fields(stage="pre_transform")
        static_transform = BaseGraphDrawingData.fields(stage="static_transform")
        transform_fields = BaseGraphDrawingData.fields(stage="transform")
        dynamic_transform_fields = BaseGraphDrawingData.fields(stage="dynamic_transform")

        self.assertTrue(True)  # TODO: add automated tests

        return

    def test_init(self):
        data_list = self.data_list
        batch = self.batch

        G = batch.G
        num_nodes = batch.num_nodes
        num_graphs = batch.num_graphs

        self.assertTrue(True)  # TODO: add automated tests

        return

    def test_pre_transform(self):
        data_list = self.data_list = [
            data.pre_transform()
            for data in self.data_list
        ]
        batch = self.batch

        # ----------------------------------------- static
        perm_index = batch.perm_index
        edge_metaindex = batch.edge_metaindex
        apsp_attr = batch.apsp_attr
        perm_weight = batch.perm_weight
        laplacian_eigenvector_pe = batch.laplacian_eigenvector_pe

        # ---------------------------------------- dynamic
        x = batch.x
        perm_attr = batch.perm_attr
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        edge_weight = batch.edge_weight

        self.assertTrue(True)  # TODO: add automated tests

        return

    def test_static_transform(self):
        data_list = self.data_list = [
            data.pre_transform().static_transform()
            for data in self.data_list
        ]
        batch = self.batch

        # ----------------------------------------- static
        name = batch.name
        dataset = batch.dataset
        n = batch.n
        m = batch.m
        edge_pair_metaindex = batch.edge_pair_metaindex

        # ---------------------------------------- dynamic
        edge_pair_index = batch.edge_pair_index

        self.assertTrue(True)  # TODO: add automated tests

        return

    def test_transform(self):
        data_list = self.data_list = [
            data.pre_transform().static_transform().dynamic_transform()
            for data in self.data_list
        ]
        batch = self.batch

        # ----------------------------------------- static
        aggr_metaindex = batch.aggr_metaindex
        pos = batch.pos

    # ---------------------------------------- dynamic
        aggr_index = batch.aggr_index
        aggr_attr = batch.aggr_attr
        aggr_weight = batch.aggr_weight

        self.assertTrue(True)  # TODO: add automated tests

        return

    def test_dynamic_transform(self):
        data_list = self.data_list = [
            data.pre_transform().static_transform().dynamic_transform().dynamic_transform()
            for data in self.data_list
        ]
        batch = self.batch

        # ----------------------------------------- static
        face = batch.face
        gabriel_index = batch.gabriel_index
        rng_index = batch.rng_index

        self.assertTrue(True)  # TODO: add automated tests

        return

    def test_new(self):
        data_list = self.data_list = [
            BaseGraphDrawingData.new(G)
            for G in self.G_list
        ]
        batch = self.batch

        self.assertTrue(True)  # TODO: add automated tests

        return

    def test_new_disconnected_graph(self):
        G = nx.Graph([
            (0, 1),
            (2, 3)
        ])
        data = BaseGraphDrawingData.new(G)

        self.assertIsNone(data)

        return


if __name__ == '__main__':
    unittest.main()
