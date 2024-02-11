import os
import random
from collections import defaultdict

import numpy as np
import networkx as nx
import torch
import torch_geometric as pyg
import torch_scatter
from tqdm.auto import *

from deepgd.model_old import DeepGD, Stress
from deepgd.data import ScalableGraphDataUniformSample
from deepgd.datasets import SuiteSparseDataset

device = 'cpu'
lr = 0.001
landmarks = 20
rand_edges = 20
batch_size = 2

torch.random.manual_seed(12345)
dataset = SuiteSparseDataset(
    min_nodes=0,
    max_nodes=7500,
    limit=1000,
    datatype=ScalableGraphDataUniformSample,
    datatype_args=dict(
        device=device,
        landmarks=20,
        random_edges=20
    )
)
shuffled_dataset, perm_idx = dataset.shuffle(return_perm=True)
len(shuffled_dataset), perm_idx

train_loader = pyg.loader.DataLoader(shuffled_dataset[:550], batch_size=batch_size, shuffle=True)
val_loader = pyg.loader.DataLoader(shuffled_dataset[550:], batch_size=batch_size, shuffle=False)

next(iter(train_loader))
