from smartgd.common.data.graph_drawing_data import *

import unittest

import networkx as nx


class TestBaseGraphDrawingData(unittest.TestCase):

    G_list:     list[nx.Graph]
    data_list:  list[GraphDrawingData]
    batch:      Batch  # Batch[GraphDrawingData]
    wheel:      GraphDrawingData
    ladder:     GraphDrawingData
    grid:       GraphDrawingData
    lollipop:   GraphDrawingData

    def setUp(self) -> None:
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
        self.data_list = list(map(GraphDrawingData, self.G_list))
        (self.wheel,
         self.ladder,
         self.grid,
         self.lollipop) = self.data_list

    @property
    def batch(self) -> Batch:
        return Batch.from_data_list(self.data_list)

    def test_struct(self):
        data_list = self.data_list = [
            data.pre_transform().dynamic_transform().post_transform("pos")
            for data in self.data_list
        ]
        batch = self.batch

        struct_1 = batch.struct(post_transform=["face", "rng_index"])
        struct_2 = batch.struct(struct_1, post_transform=["face", "rng_index"])

        self.assertTrue(True)  # TODO: add automated tests

        return


if __name__ == '__main__':
    unittest.main()
