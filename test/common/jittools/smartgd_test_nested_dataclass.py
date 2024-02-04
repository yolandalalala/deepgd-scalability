import inspect

import smartgd

from dataclasses import dataclass, field
from typing import Callable, Union, Optional
from torch import nn, jit
import torch


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

