import inspect

import smartgd

from dataclasses import dataclass, field
from typing import Callable, Union, Optional
from torch import nn, jit
import torch

# %%


# @jittable
# @dataclass(kw_only=True)
# class NestedModel(nn.Module):
#     a: int = 1
#
#     def __post_init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return x + self.a
#
#
# @jittable
# @dataclass(kw_only=True, eq=False)
# class Model(nn.Module):
#
#     @dataclass(kw_only=True, frozen=True)
#     class Config:
#         width: int
#         depth: int
#         dims: list
#         hidden_act: str
#         out_act: Optional[str]
#         bn: Optional[str]
#         dp: float
#         residual: bool
#
#     alpha: int = 1
#     config: Config = Config(
#         width=0,
#         depth=0,
#         dims=[],
#         hidden_act="leaky_relu",
#         out_act="tanh",
#         bn="batch_norm",
#         dp=0.0,
#         residual=False
#     )
#
#     def __post_init__(self):
#         super().__init__()
#         self.model = NestedModel()
#
#     def forward(self, x):
#         return self.model(x) + self.alpha


# %%

# model = Model()

# %%

# model in set()

# %%

# print(jit.script(model))


# %%

@jit.script
@smartgd.jittable
@dataclass(kw_only=True, eq=False, repr=False)
class Outer:

    @smartgd.jittable
    @dataclass(kw_only=True, eq=False, repr=False)
    class Inner:
        a: int

    b: int
    inner: Inner


def method(self, x=0):
    return x


Outer.method = method


@smartgd.jittable
@dataclass(kw_only=True)
class Model(nn.Module):

    alpha: int = 1

    def __post_init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, outer: Outer):
        return torch.stack([
            torch.tensor(self.alpha),
            x,
            torch.tensor(outer.b),
            torch.tensor(outer.inner.a)
        ])


script = jit.script(Model())

print(script)
print(script(torch.tensor(1), Outer(b=1, inner=Outer.Inner(a=1))))

print(Outer.method)
print(Outer(b=1, inner=Outer.Inner(a=1)).method)

