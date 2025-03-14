import functools
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    get_args,
    overload,
)

import numpy as np
import torch
import torch.utils._pytree as pytree
from torch import Tensor

import torch_geometric.typing
from torch_geometric import Index, is_compiling
from torch_geometric.index import index2ptr, ptr2index
from torch_geometric.typing import INDEX_DTYPES, SparseTensor

from torch_geometric.edge_index import (set_tuple_item, EdgeIndex, maybe_add, maybe_sub)

aten = torch.ops.aten

HANDLED_FUNCTIONS: Dict[Callable, Callable] = {}


class SortOrder(Enum):
    ID = 'id'
    DIST = 'dist'


class CatMetadata(NamedTuple):
    dim_size: List[Optional[int]]
    sort_order: List[Optional[SortOrder]]


def implements(torch_function: Callable) -> Callable:
    r"""Registers a :pytorch:`PyTorch` function override."""

    @functools.wraps(torch_function)
    def decorator(my_function: Callable) -> Callable:
        HANDLED_FUNCTIONS[torch_function] = my_function
        return my_function

    return decorator


def assert_valid_dtype(tensor: Tensor) -> None:
    if tensor.dtype not in INDEX_DTYPES:
        raise ValueError(f"'SourceIndex' holds an unsupported data type "
                         f"(got '{tensor.dtype}', but expected one of "
                         f"{INDEX_DTYPES})")


def assert_two_dimensional(tensor: Tensor) -> None:
    if tensor.dim() != 2:
        raise ValueError(f"'SourceIndex' needs to be two-dimensional "
                         f"(got {tensor.dim()} dimensions)")


def assert_contiguous(tensor: Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError("'SourceIndex' needs to be contiguous. Please call "
                         "`edge_index.contiguous()` before proceeding.")


def assert_sorted(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self: 'SourceIndex', *args: Any, **kwargs: Any) -> Any:
        if not self.is_sorted:
            cls_name = self.__class__.__name__
            raise ValueError(
                f"Cannot call '{func.__name__}' since '{cls_name}' is not "
                f"sorted. Please call `{cls_name}.sort_by(...)` first.")
        return func(self, *args, **kwargs)

    return wrapper


class SourceIndex(Tensor):
    r"""A KN :obj:`edge_index` tensor with additional (meta)data attached.

    :class:`SourceIndex` is a :pytorch:`null` :class:`torch.Tensor`, that holds
    an :obj:`edge_index` representation of shape :obj:`[num_source_nodes, K]`.
    The source indices represent connections to target nodes.
    """
    # See "https://pytorch.org/docs/stable/notes/extending.html"
    # for a basic tutorial on how to subclass `torch.Tensor`.

    # The underlying tensor representation:
    _data: Tensor

    # The size of the dimension (number of source nodes):
    _dim_size: Optional[int] = None

    # Whether the `edge_index` representation is non-sorted (`None`), or sorted
    # based on distances or indices.
    _sort_order: Optional[SortOrder] = None

    # Whenever we perform a concatenation of edge indices, we cache the
    # original metadata to be able to reconstruct individual edge indices:
    _cat_metadata: Optional[CatMetadata] = None

    @staticmethod
    def __new__(
            cls: Type,
            data: Any,
            *args: Any,
            dim_size: Optional[int] = None,
            sparse_size: Optional[Tuple[Optional[int], Optional[int]]] = None,
            sort_order: Optional[Union[str, SortOrder]] = None,
            **kwargs: Any,
    ) -> 'SourceIndex':
        if not isinstance(data, Tensor):
            data = torch.tensor(data, *args, **kwargs)
        elif len(args) > 0:
            raise TypeError(
                f"new() received an invalid combination of arguments - got "
                f"(Tensor, {', '.join(str(type(arg)) for arg in args)})")
        elif len(kwargs) > 0:
            raise TypeError(f"new() received invalid keyword arguments - got "
                            f"{set(kwargs.keys())})")

        assert isinstance(data, Tensor)

        if isinstance(data, cls):  # If passed `SourceIndex`, inherit metadata:
            dim_size = dim_size or data.dim_size
            sort_order = sort_order or data.sort_order

        # For backward compatibility, handle sparse_size if provided
        if sparse_size is not None and dim_size is None:
            dim_size = sparse_size[0]
            if sparse_size[1] is not None and sparse_size[1] != data.size(0):
                import warnings
                warnings.warn(
                    f"Inconsistent number of source nodes in sparse_size "
                    f"(got {sparse_size[1]}, but tensor has {data.size(0)})")

        # Convert `torch.sparse` tensors to `SourceIndex` representation:
        if data.layout == torch.sparse_coo:
            raise NotImplementedError("Cannot automatically convert torch.sparse to SourceIndex")
        if data.layout == torch.sparse_csr:
            raise NotImplementedError("Cannot automatically convert torch.sparse to SourceIndex")
        if (torch_geometric.typing.WITH_PT112 and data.layout == torch.sparse_csc):
            raise NotImplementedError("Cannot automatically convert torch.sparse to SourceIndex")

        assert_valid_dtype(data)
        assert_two_dimensional(data)
        assert_contiguous(data)

        out = Tensor._make_wrapper_subclass(  # type: ignore
            cls,
            size=data.size(),
            strides=data.stride(),
            dtype=data.dtype,
            device=data.device,
            layout=data.layout,
            requires_grad=False,
        )
        assert isinstance(out, SourceIndex)

        # Attach metadata:
        out._data = data
        out._dim_size = dim_size
        out._sort_order = None if sort_order is None else SortOrder(sort_order)

        if isinstance(data, cls):  # If passed `SourceIndex`, inherit metadata:
            out._data = data._data

        return out

    # Validation ##############################################################

    def validate(self) -> 'SourceIndex':
        r"""Validates the :class:`SourceIndex` representation.

        In particular, it ensures that

        * it only holds valid indices.
        * the sort order is correctly set.
        * indices are consistent with the dimension size if specified.
        """
        assert_valid_dtype(self._data)
        assert_two_dimensional(self._data)
        assert_contiguous(self._data)

        if self.numel() > 0 and self._data.min() < 0:
            raise ValueError(f"'{self.__class__.__name__}' contains negative "
                             f"indices (got {int(self.min())})")

        if (self.numel() > 0 and self._dim_size is not None
                and self._data.max() >= self._dim_size):
            raise ValueError(f"'{self.__class__.__name__}' contains larger "
                             f"indices than its dimension size "
                             f"(got {int(self._data.max())}, but expected "
                             f"values smaller than {self._dim_size})")

        if self.is_sorted_by_id and (self._data.diff(dim=-1) < 0).any():
            raise ValueError(f"'{self.__class__.__name__}' is not sorted by "
                             f"source indices")

        if self.is_sorted_by_distance and True:
            # fixme: not implemented, maybe not possible to check?
            pass

        return self

    # Properties ##############################################################

    @property
    def dim_size(self) -> Optional[int]:
        r"""The dimension size (number of source nodes)."""
        return self._dim_size

    @property
    def num_target_nodes(self) -> Optional[int]:
        r"""The number of target nodes."""
        return self._data.size(0)

    @property
    def num_source_nodes(self) -> int:
        r"""The number of source nodes."""
        return self._dim_size

    @property
    def k(self) -> int:
        r"""The number of source nodes per row."""
        return self._data.size(1)

    @property
    def sort_order(self) -> Optional[str]:
        r"""The sort order of indices, either :obj:`"id"`, :obj:`"dist"` or
        :obj:`None`.
        """
        return None if self._sort_order is None else self._sort_order.value

    @property
    def is_sorted(self) -> bool:
        r"""Returns whether indices are either sorted by rows or columns."""
        return self._sort_order is not None

    @property
    def is_sorted_by_distance(self) -> bool:
        r"""Returns whether indices are sorted by distance."""
        return self._sort_order == SortOrder.DIST

    @property
    def is_sorted_by_id(self) -> bool:
        r"""Returns whether indices are sorted by id."""
        return self._sort_order == SortOrder.ID

    @property
    def dtype(self) -> torch.dtype:  # type: ignore
        # TODO Remove once PyTorch does not override `dtype` in `DataLoader`.
        return self._data.dtype

    # For backward compatibility
    @property
    def sparse_size(self) -> Tuple[Optional[int], int]:
        r"""The size of the underlying sparse matrix (deprecated).
        Use dim_size and num_source_nodes instead.
        """
        # todo: maybe remove this?
        # import warnings
        # warnings.warn(
        #     "sparse_size is deprecated, use dim_size and num_source_nodes instead",
        #     DeprecationWarning, stacklevel=2
        # )
        return (self.num_source_nodes, self.num_target_nodes)

    # Cache Interface #########################################################

    def get_dim_size(self) -> int:
        r"""The dimension size (number of source nodes).
        Automatically computed and cached when not explicitly set.
        """
        if self._dim_size is not None:
            return self._dim_size

        dim_size = int(self._data.max()) + 1 if self.numel() > 0 else 0
        self._dim_size = dim_size
        return dim_size

    def get_num_source_nodes(self) -> int:
        r"""The number of source nodes.
        Automatically computed and cached when not explicitly set.
        """
        return self.get_dim_size()

    # For backward compatibility
    def get_sparse_size(
            self,
            dim: Optional[int] = None,
    ) -> Union[torch.Size, int]:
        r"""The size of the underlying sparse matrix (deprecated).
        Use get_dim_size() and num_source_nodes instead.
        """
        import warnings
        warnings.warn(
            "get_sparse_size is deprecated, use get_dim_size and num_source_nodes instead",
            DeprecationWarning, stacklevel=2)

        if dim is not None:
            if dim == 0:
                return self.get_dim_size()
            else:
                return self.num_source_nodes

        return torch.Size((self.get_dim_size(), self.num_source_nodes))

    def get_num_rows(self) -> int:
        r"""For backward compatibility. Use get_dim_size() instead."""
        import warnings
        warnings.warn(
            "get_num_rows is deprecated, use get_dim_size instead",
            DeprecationWarning, stacklevel=2)
        return self.get_dim_size()

    def get_num_cols(self) -> int:
        r"""For backward compatibility. Use num_source_nodes instead."""
        import warnings
        warnings.warn(
            "get_num_cols is deprecated, use num_source_nodes instead",
            DeprecationWarning, stacklevel=2)
        return self.num_source_nodes

    # Methods #################################################################

    def share_memory_(self) -> 'SourceIndex':
        """"""  # noqa: D419
        self._data.share_memory_()
        return self

    def is_shared(self) -> bool:
        """"""  # noqa: D419
        return self._data.is_shared()

    def as_tensor(self) -> Tensor:
        r"""Zero-copies the :class:`SourceIndex` representation back to a
        :class:`torch.Tensor` representation.
        """
        return self._data

    def sort_by(
            self,
            sort_order: Union[str, SortOrder],
    ) -> 'SortReturnType':
        r"""Sorts the elements by id or distance.

        Args:
            sort_order (str): The sort order, either :obj:`"id"` or
                :obj:`"dist"`.
        """
        from torch_geometric.utils import index_sort

        sort_order = SortOrder(sort_order)

        if self._sort_order == sort_order:  # Nothing to do.
            return SortReturnType(self, None)

        raise NotImplementedError("SourceIndex.sort_by not yet implemented")

    def get_target_index(self) -> Tensor:
        # Create an index tensor representing target nodes
        return torch.arange(self.num_target_nodes, device=self.device, dtype=self.dtype).unsqueeze(-1).expand(
            [-1, self.k])

    def to_edge_index(self) -> EdgeIndex:

        return EdgeIndex(torch.stack([
            self.flatten(),  # todo: maybe sort last dim, for efficiency?
            self.get_target_index().flatten(),
        ], dim=0), sparse_size=(self.dim_size, self.num_target_nodes) if self.dim_size is not None else None)

    def to_sparse_tensor(self) -> SparseTensor:
        r"""Converts the :class:`SourceIndex` to a :class:`SparseTensor`.

        Returns:
            SparseTensor: The resulting sparse tensor.
        """

        # Create the SparseTensor using the row and col indices:
        sparse_tensor = SparseTensor(
            row=self.get_target_index().flatten(),
            col=self.flatten(),
            sparse_sizes=(self.num_target_nodes, self.num_source_nodes),
            is_sorted=self.is_sorted_by_id,
            trust_data=True,
        )

        return sparse_tensor

    # PyTorch/Python builtins #################################################

    def __tensor_flatten__(self) -> Tuple[List[str], Tuple[Any, ...]]:
        attrs = ['_data']

        ctx = (
            self._dim_size,
            self._sort_order,
            self._cat_metadata,
        )

        return attrs, ctx

    @staticmethod
    def __tensor_unflatten__(
            inner_tensors: Dict[str, Any],
            ctx: Tuple[Any, ...],
            outer_size: Tuple[int, ...],
            outer_stride: Tuple[int, ...],
    ) -> 'SourceIndex':
        edge_index = SourceIndex(
            inner_tensors['_data'],
            dim_size=ctx[0],
            sort_order=ctx[1],
        )

        edge_index._cat_metadata = ctx[2]

        return edge_index

    # Prevent auto-wrapping outputs back into the proper subclass type:
    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(
            cls: Type,
            func: Callable[..., Any],
            types: Iterable[Type[Any]],
            args: Iterable[Tuple[Any, ...]] = (),
            kwargs: Optional[Dict[Any, Any]] = None,
    ) -> Any:
        # `SourceIndex` should be treated as a regular PyTorch tensor for all
        # standard PyTorch functionalities. However,
        # * some of its metadata can be transferred to new functions, e.g.,
        #   `torch.cat(dim=1)` can inherit the dimension size, or
        #   `torch.narrow(dim=1)` can inherit cached information.
        # * not all operations lead to valid `SourceIndex` tensors again, e.g.,
        #   `torch.sum()` does not yield a `SourceIndex` as its output

        # To account for this, we hold a number of `HANDLED_FUNCTIONS` that
        # implement specific functions for valid `SourceIndex` routines.
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **(kwargs or {}))

        # For all other PyTorch functions, we treat them as vanilla tensors.
        args = pytree.tree_map_only(SourceIndex, lambda x: x._data, args)
        if kwargs is not None:
            kwargs = pytree.tree_map_only(SourceIndex, lambda x: x._data, kwargs)
        return func(*args, **(kwargs or {}))

    def __repr__(self) -> str:  # type: ignore
        prefix = f'{self.__class__.__name__}('
        indent = len(prefix)
        tensor_str = torch._tensor_str._tensor_str(self._data, indent)

        suffixes = []
        if self._dim_size is not None:
            suffixes.append(f'dim_size={self._dim_size}')
        if (self.device.type != torch._C._get_default_device()
                or (self.device.type == 'cuda'
                    and torch.cuda.current_device() != self.device.index)
                or (self.device.type == 'mps')):
            suffixes.append(f"device='{self.device}'")
        if self.dtype != torch.int64:
            suffixes.append(f'dtype={self.dtype}')
        if self.is_sorted:
            suffixes.append(f'sort_order={self.sort_order}')

        return torch._tensor_str._add_suffixes(prefix + tensor_str, suffixes,
                                               indent, force_newline=False)

    def tolist(self) -> List[Any]:
        """"""  # noqa: D419
        return self._data.tolist()

    def numpy(self, *, force: bool = False) -> np.ndarray:
        """"""  # noqa: D419
        return self._data.numpy(force=force)

    # Helpers #################################################################

    def _shallow_copy(self) -> 'SourceIndex':
        out = SourceIndex(self._data)
        out._dim_size = self._dim_size
        out._sort_order = self._sort_order
        out._cat_metadata = self._cat_metadata
        return out

    def _clear_metadata(self) -> 'SourceIndex':
        self._dim_size = None
        self._sort_order = None
        self._cat_metadata = None
        return self


class SortReturnType(NamedTuple):
    values: SourceIndex
    indices: Optional[Tensor]


def apply_(
        tensor: SourceIndex,
        fn: Callable,
        *args: Any,
        **kwargs: Any,
) -> Union[SourceIndex, Tensor]:
    data = fn(tensor._data, *args, **kwargs)

    if data.dtype not in INDEX_DTYPES:
        return data

    if tensor._data.data_ptr() != data.data_ptr():
        out = SourceIndex(data)
    else:  # In-place:
        tensor._data = data
        out = tensor

    # Copy metadata:
    out._dim_size = tensor._dim_size
    out._sort_order = tensor._sort_order
    out._cat_metadata = tensor._cat_metadata

    return out


@implements(aten.clone.default)
def _clone(
        tensor: SourceIndex,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
) -> SourceIndex:
    out = apply_(tensor, aten.clone.default, memory_format=memory_format)
    assert isinstance(out, SourceIndex)
    return out


@implements(aten._to_copy.default)
def _to_copy(
        tensor: SourceIndex,
        *,
        dtype: Optional[torch.dtype] = None,
        layout: Optional[torch.layout] = None,
        device: Optional[torch.device] = None,
        pin_memory: bool = False,
        non_blocking: bool = False,
        memory_format: Optional[torch.memory_format] = None,
) -> Union[SourceIndex, Tensor]:
    return apply_(
        tensor,
        aten._to_copy.default,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        memory_format=memory_format,
    )


@implements(aten.alias.default)
def _alias(tensor: SourceIndex) -> SourceIndex:
    return tensor._shallow_copy()


@implements(aten._pin_memory.default)
def _pin_memory(tensor: SourceIndex) -> SourceIndex:
    out = apply_(tensor, aten._pin_memory.default)
    assert isinstance(out, SourceIndex)
    return out


@implements(aten.cat.default)
def _cat(
        tensors: List[Union[SourceIndex, Tensor]],
        dim: int = 0,
) -> Union[SourceIndex, Tensor]:
    data_list = pytree.tree_map_only(SourceIndex, lambda x: x._data, tensors)
    data = aten.cat.default(data_list, dim=dim)

    if dim not in [0, 1, -1, -2]:  # No valid `SourceIndex` anymore.
        return data

    if any([not isinstance(tensor, SourceIndex) for tensor in tensors]):
        return data

    out = SourceIndex(data)

    dim_size_list = [t.dim_size for t in tensors]  # type: ignore
    sort_order_list = [t._sort_order for t in tensors]  # type: ignore

    # Find the maximum dimension size
    max_dim_size: Optional[int] = 0
    for size in dim_size_list:
        if size is None:
            max_dim_size = None
            break
        max_dim_size = max(size, max_dim_size)

    out._dim_size = max_dim_size
    out._cat_metadata = CatMetadata(
        dim_size=dim_size_list,
        sort_order=sort_order_list,
    )

    return out


@implements(aten.index_select.default)
def _index_select(
        input: SourceIndex,
        dim: int,
        index: Tensor,
) -> Union[SourceIndex, Tensor]:
    out = aten.index_select.default(input._data, dim, index)

    if len(out.shape) == 2:
        # Indexing produces a valid SourceIndex as long as the output is still 2D
        out = SourceIndex(out, dim_size=input.dim_size)

    # todo: taking a row or column should return a (1-d) Index
    return out


@implements(aten.slice.Tensor)
def _slice(
        input: SourceIndex,
        dim: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: int = 1,
) -> Union[SourceIndex, Tensor]:
    if ((start is None or start == 0 or start <= -input.size(dim))
            and (end is None or end > input.size(dim)) and step == 1):
        return input._shallow_copy()  # No-op.

    out = aten.slice.Tensor(input._data, dim, start, end, step)

    if len(out.shape) == 2:
        if step != 1:
            out = out.contiguous()

        out = SourceIndex(out, dim_size=input.dim_size)
        out._sort_order = input._sort_order

    return out


@implements(aten.index.Tensor)
def _index(
        input: Union[SourceIndex, Tensor],
        indices: List[Optional[Union[Tensor, SourceIndex]]],
) -> Union[SourceIndex, Tensor]:
    if not isinstance(input, SourceIndex):
        indices = pytree.tree_map_only(SourceIndex, lambda x: x._data, indices)
        return aten.index.Tensor(input, indices)

    out = aten.index.Tensor(input._data, indices)

    if len(out.shape) != 2:
        return out

    out = SourceIndex(out, dim_size=input.dim_size)

    return out


@implements(aten.select.int)
def _select(input: SourceIndex, dim: int, index: int) -> Union[Tensor, Index]:
    out = aten.select.int(input._data, dim, index)

    # Indexing along dimensions produces a valid Index
    out = Index(out)

    if dim in [0, -2]:
        out._dim_size = input.dim_size
        out._is_sorted = input.is_sorted_by_id
    else:
        assert dim in [1, -1]
        # When selecting along the K dimension, we can't infer a dim_size

    return out


@implements(aten.unbind.int)
def _unbind(
        input: SourceIndex,
        dim: int = 0,
) -> Union[List[Index], List[Tensor]]:
    raise NotImplementedError("unbind not implemented for SourceIndex")


@implements(aten.add.Tensor)
def _add(
        input: Union[int, Tensor, SourceIndex],
        other: Union[int, Tensor, SourceIndex],
        *,
        alpha: int = 1,
) -> Union[SourceIndex, Tensor]:
    dim_size = input.dim_size if isinstance(input, SourceIndex) else other.dim_size

    out = aten.add.Tensor(
        input._data if isinstance(input, SourceIndex) else input,
        other._data if isinstance(other, SourceIndex) else other,
        alpha=alpha,
    )

    if out.dtype not in INDEX_DTYPES:
        return out
    if out.shape != input.shape:
        return out

    out = SourceIndex(out)

    if isinstance(other, Tensor) and other.numel() <= 1:
        other = int(other)

    total_dim_size: Optional[int] = dim_size
    if isinstance(other, int) and dim_size is not None:
        total_dim_size = dim_size + (other * alpha)
    elif isinstance(other, SourceIndex):
        raise NotImplementedError('Additions between SourceIndex are not implemented')

    out._dim_size = total_dim_size
    return out


@implements(aten.add_.Tensor)
def add_(
        input: SourceIndex,
        other: Union[int, Tensor, SourceIndex],
        *,
        alpha: int = 1,
) -> SourceIndex:
    dim_size = input._dim_size

    aten.add_.Tensor(
        input._data,
        other._data if isinstance(other, SourceIndex) else other,
        alpha=alpha,
    )

    if isinstance(other, Tensor) and other.numel() <= 1:
        other = int(other)

    if isinstance(other, int):
        if dim_size is not None:
            input._dim_size = dim_size + (other * alpha)
    elif isinstance(other, SourceIndex):
        if dim_size is not None and other.dim_size is not None:
            input._dim_size = max(dim_size, other.dim_size)

    return input


@implements(aten.sub.Tensor)
def _sub(
        input: SourceIndex,
        other: Union[int, Tensor, SourceIndex],
        *,
        alpha: int = 1,
) -> Union[SourceIndex, Tensor]:
    dim_size = input.dim_size if isinstance(input, SourceIndex) else other.dim_size

    out = aten.sub.Tensor(
        input._data if isinstance(input, SourceIndex) else input,
        other._data if isinstance(other, SourceIndex) else other,
        alpha=alpha,
    )

    if out.dtype not in INDEX_DTYPES:
        return out
    if out.shape != input.shape:
        return out

    out = SourceIndex(out)

    if isinstance(other, Tensor) and other.numel() <= 1:
        other = int(other)

    total_dim_size: Optional[int] = dim_size
    if isinstance(other, int) and dim_size is not None:
        total_dim_size = max(0, dim_size - (other * alpha))
    elif isinstance(other, SourceIndex):
        raise NotImplementedError('Subtractions between SourceIndex are not implemented')

    out._dim_size = total_dim_size
    return out


@implements(aten.sub_.Tensor)
def sub_(
        input: SourceIndex,
        other: Union[int, Tensor, SourceIndex],
        *,
        alpha: int = 1,
) -> SourceIndex:
    dim_size = input._dim_size
    sort_order = input._sort_order
    input._clear_metadata()

    aten.sub_.Tensor(
        input._data,
        other._data if isinstance(other, SourceIndex) else other,
        alpha=alpha,
    )

    if isinstance(other, Tensor) and other.numel() <= 1:
        other = int(other)

    if isinstance(other, int):
        if dim_size is not None:
            input._dim_size = max(0, dim_size - (other * alpha))
        input._sort_order = sort_order

    return input


@implements(aten.detach.default)
def detach(self: SourceIndex):
    return SourceIndex(
        self._data.detach(),
        dim_size=self._dim_size,
        sort_order=self._sort_order,
    )


@implements(aten.alias.default)
def alias(self: SourceIndex):
    return SourceIndex(
        self._data.alias(),
        dim_size=self._dim_size,
        sort_order=self._sort_order,
    )
