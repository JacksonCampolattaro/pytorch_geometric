from typing import Optional, Union

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric import SourceIndex
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    sort_edge_index,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)


@functional_transform('to_sparse_tensor')
class ToSparseTensor(BaseTransform):
    r"""Converts the :obj:`edge_index` attributes of a homogeneous or
    heterogeneous data object into a **transposed**
    :class:`torch_sparse.SparseTensor` or :pytorch:`PyTorch`
    :class:`torch.sparse.Tensor` object with key :obj:`adj_t`
    (functional name: :obj:`to_sparse_tensor`).

    .. note::

        In case of composing multiple transforms, it is best to convert the
        :obj:`data` object via :class:`ToSparseTensor` as late as possible,
        since there exist some transforms that are only able to operate on
        :obj:`data.edge_index` for now.

    Args:
        attr (str, optional): The name of the attribute to add as a value to
            the :class:`~torch_sparse.SparseTensor` or
            :class:`torch.sparse.Tensor` object (if present).
            (default: :obj:`edge_weight`)
        remove_edge_index (bool, optional): If set to :obj:`False`, the
            :obj:`edge_index` tensor will not be removed.
            (default: :obj:`True`)
        fill_cache (bool, optional): If set to :obj:`True`, will fill the
            underlying :class:`torch_sparse.SparseTensor` cache (if used).
            (default: :obj:`True`)
        layout (torch.layout, optional): Specifies the layout of the returned
            sparse tensor (:obj:`None`, :obj:`torch.sparse_coo` or
            :obj:`torch.sparse_csr`).
            If set to :obj:`None` and the :obj:`torch_sparse` dependency is
            installed, will convert :obj:`edge_index` into a
            :class:`torch_sparse.SparseTensor` object.
            If set to :obj:`None` and the :obj:`torch_sparse` dependency is
            not installed, will convert :obj:`edge_index` into a
            :class:`torch.sparse.Tensor` object with layout
            :obj:`torch.sparse_csr`. (default: :obj:`None`)
        replace_edge_index (bool, optional): If set to :obj:`True`, the result
            will be saved back to :obj:`edge_index` instead of creating a new
            :obj:`adj_t` attribute. If :obj:`True`, this will override
            :obj:`remove_edge_index` to :obj:`True`.
            (default: :obj:`False`)
    """

    def __init__(
            self,
            attr: Optional[str] = 'edge_weight',
            remove_edge_index: bool = True,
            fill_cache: bool = True,
            layout: Optional[int] = None,
            replace_edge_index: bool = False,
    ) -> None:
        if layout not in {None, torch.sparse_coo, torch.sparse_csr}:
            raise ValueError(f"Unexpected sparse tensor layout "
                             f"(got '{layout}')")

        self.attr = attr
        self.remove_edge_index = remove_edge_index
        self.fill_cache = fill_cache
        self.layout = layout
        self.replace_edge_index = replace_edge_index

    def forward(
            self,
            data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        for store in data.edge_stores:
            # Check for either edge_index or source_index attributes
            has_edge_index = 'edge_index' in store

            if not has_edge_index:
                continue

            keys, values = [], []
            for key, value in store.items():
                if key in {'edge_index', 'source_index', 'edge_label', 'edge_label_index'}:
                    continue

                if store.is_edge_attr(key):
                    keys.append(key)
                    values.append(value)

            # Process based on which index type we have
            size = store.size()[::-1]
            edge_weight: Optional[Tensor] = None
            if self.attr is not None and self.attr in store:
                edge_weight = store[self.attr]

            if isinstance(store.edge_index, SourceIndex):
                # fixme: add support for other layouts; edge attributes
                sparse_tensor = store.edge_index.to_sparse_tensor()
            else:
                # Handle regular edge_index
                edge_index = store.edge_index

                # Sort the edge_index and associated values
                edge_index, values = sort_edge_index(
                    edge_index,
                    values,
                    sort_by_row=False,
                )

                for key, value in zip(keys, values):
                    store[key] = value

                # Create sparse tensor based on layout preference
                layout = self.layout
                sparse_tensor = None

                if layout is None and torch_geometric.typing.WITH_TORCH_SPARSE:
                    sparse_tensor = SparseTensor(
                        row=edge_index[1],
                        col=edge_index[0],
                        value=edge_weight,
                        sparse_sizes=size,
                        is_sorted=True,
                        trust_data=True,
                    )

                # TODO Multi-dimensional edge attributes only supported for COO.
                elif ((edge_weight is not None and edge_weight.dim() > 1)
                      or layout == torch.sparse_coo):
                    assert size[0] is not None and size[1] is not None
                    sparse_tensor = to_torch_coo_tensor(
                        edge_index.flip([0]),
                        edge_attr=edge_weight,
                        size=size,
                    )

                elif layout is None or layout == torch.sparse_csr:
                    assert size[0] is not None and size[1] is not None
                    sparse_tensor = to_torch_csr_tensor(
                        edge_index.flip([0]),
                        edge_attr=edge_weight,
                        size=size,
                    )

            # Decide where to store the result
            if self.replace_edge_index:
                # Store back to edge_index and ensure it's removed from other places
                store['edge_index'] = sparse_tensor

                # Remove adj_t if it exists
                if 'adj_t' in store:
                    del store['adj_t']
            else:
                # Store as adj_t (traditional behavior)
                store.adj_t = sparse_tensor

                # Handle removal of original indices if needed
                if self.remove_edge_index:
                    if has_edge_index:
                        del store['edge_index']

                    if self.attr is not None and self.attr in store:
                        del store[self.attr]

            # Handle caching
            if self.fill_cache and isinstance(sparse_tensor, SparseTensor):
                # Pre-process some important attributes.
                sparse_tensor.storage.rowptr()
                sparse_tensor.storage.csr2csc()

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(attr={self.attr}, '
                f'replace_edge_index={self.replace_edge_index}, '
                f'layout={self.layout})')
