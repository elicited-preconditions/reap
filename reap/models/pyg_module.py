# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from abc import ABC, abstractmethod
from functools import singledispatchmethod
from typing import Dict, Optional, Union
import mifrost
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.typing import Adj

from reap.models.mixins import DeviceAwareMixin


class PyGModule(DeviceAwareMixin, Module, ABC):
    @abstractmethod
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor = None,
        **kwargs,
    ) -> Tensor: ...

    @singledispatchmethod
    def __call__(self, *args, **kwargs):
        if args:
            msg = f"Invalid input type {args[0].__class__} for '__call__'"
        else:
            msg = "No input for '__call__'"
        raise NotImplementedError(msg)

    @__call__.register(dict)
    def _(self, x, edge_index, batch=None, *args, **kwargs):
        return super().__call__(x, edge_index, batch, *args, **kwargs)

    @__call__.register(Data)
    @__call__.register(Batch)
    def _(self, data, *args, **kwargs):
        return super().__call__(*self.unpack(data), *args, **kwargs)

    @__call__.register(mifrost.BatchEncoding)
    def _(self, data, *args, **kwargs):
        return super().__call__(*self.unpack_native(data), *args, **kwargs)

    @__call__.register(mifrost.HomoBatchEncodingView)
    def _(self, data, *args, **kwargs):
        return super().__call__(*self.unpack_native(data), *args, **kwargs)

    def unpack_native(
        self, data: mifrost.BatchEncoding | mifrost.HomoBatchEncodingView
    ):
        if isinstance(data, mifrost.BatchEncoding):
            data = data.as_homo()
        return (
            data.x.to(self.device),
            data.edge_index.to(self.device),
            data.batch.to(self.device),
        )

    @classmethod
    def unpack(
        cls,
        data: Union[Data, Batch, mifrost.BatchEncoding, mifrost.HomoBatchEncodingView],
    ):
        if isinstance(data, mifrost.BatchEncoding):
            data = data.as_homo()
        if isinstance(data, mifrost.HomoBatchEncodingView):
            return data.x, data.edge_index, data.batch
        return data.x, data.edge_index, data.batch


class PyGHeteroModule(DeviceAwareMixin, Module, ABC):
    @abstractmethod
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
        **kwargs,
    ): ...

    @singledispatchmethod
    def __call__(self, *args):
        if args:
            msg = f"Invalid input type {args[0].__class__} for '__call__'"
        else:
            msg = "No input for '__call__'"
        raise NotImplementedError(msg)

    @__call__.register(dict)
    def _(self, x_dict, edge_index_dict, batch_dict=None, *args, **kwargs):
        return super().__call__(x_dict, edge_index_dict, batch_dict, *args, **kwargs)

    @__call__.register(Batch)
    def _(self, data, *args, **kwargs):
        return super().__call__(*self.unpack(data), *args, **kwargs)

    @__call__.register(HeteroData)
    def _(self, data, *args, **kwargs):
        return super().__call__(
            data.x_dict, data.edge_index_dict, *args, batch=None, **kwargs
        )

    @__call__.register(mifrost.BatchEncoding)
    def _(self, data, *args, **kwargs):
        return super().__call__(*self.unpack_native(data), *args, **kwargs)

    @__call__.register(mifrost.HeteroBatchEncodingView)
    def _(self, data, *args, **kwargs):
        return super().__call__(*self.unpack_native(data), *args, **kwargs)

    def unpack_native(
        self,
        data: mifrost.BatchEncoding | mifrost.HeteroBatchEncodingView,
    ):
        if isinstance(data, mifrost.BatchEncoding):
            data = data.as_hetero()
        data = data.to(self.device)
        x_dict = {key: value.to(self.device) for key, value in data.x_dict.items()}
        edge_index_dict = {
            key: value.to(self.device) for key, value in data.edge_index_dict.items()
        }
        batch_dict = {
            key: value.to(self.device) for key, value in data.batch_dict.items()
        }
        return x_dict, edge_index_dict, batch_dict

    @classmethod
    def unpack(
        cls,
        data: Union[
            HeteroData,
            Batch,
            mifrost.HeteroBatchEncodingView,
        ],
    ):
        return data.x_dict, data.edge_index_dict, data.batch_dict
