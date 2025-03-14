import os.path as osp
from typing import List

import numpy as np
import pytest
import torch
from torch import tensor

import torch_geometric.typing
from torch_geometric import SourceIndex, Index
from torch_geometric.data import Data
from torch_geometric.io import fs
from torch_geometric.testing import onlyCUDA, withCUDA
from torch_geometric.typing import INDEX_DTYPES

DTYPES = [pytest.param(dtype, id=str(dtype)[6:]) for dtype in INDEX_DTYPES]


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_basic(dtype, device):
    kwargs = dict(dtype=dtype, device=device, sparse_size=(3, 4))
    index = SourceIndex([[0], [1], [1], [2]], **kwargs)
    index.validate()
    assert isinstance(index, SourceIndex)

    assert str(index).startswith('SourceIndex([[0],\n             [1],\n             [1],\n             [2]],')
    assert (f"device='{device}'" in str(index)) == index.is_cuda
    assert (f'dtype={dtype}' in str(index)) == (dtype != torch.long)

    assert index.dtype == dtype
    assert index.device == device
    assert index.sparse_size == (3, 4)

    out = index.as_tensor()
    assert not isinstance(out, SourceIndex)
    assert out.dtype == dtype
    assert out.device == device

    out = index * 1
    assert not isinstance(out, SourceIndex)
    assert out.dtype == dtype
    assert out.device == device


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_identity(dtype, device):
    raise NotImplementedError("This test of SourceIndex has not yet been implemented")


def test_validate():
    raise NotImplementedError("This test of SourceIndex has not yet been implemented")


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_clone(dtype, device):
    kwargs = dict(dtype=dtype, device=device, sparse_size=(3, 4))
    index = SourceIndex([[0], [1], [1], [2]], **kwargs)

    out = index.clone()
    assert isinstance(out, SourceIndex)
    assert out.dtype == dtype
    assert out.device == device
    assert out.sparse_size == (3, 4)

    out = torch.clone(index)
    assert isinstance(out, SourceIndex)
    assert out.dtype == dtype
    assert out.device == device
    assert out.sparse_size == (3, 4)


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_to_function(dtype, device):
    kwargs = dict(dtype=dtype, device=device, sparse_size=(3, 4))
    index = SourceIndex([[0], [1], [1], [2]], **kwargs)

    index = index.to(device)
    assert isinstance(index, SourceIndex)
    assert index.device == device

    out = index.to(device='cpu')
    assert isinstance(out, SourceIndex)
    assert out.device == torch.device('cpu')

    out = index.detach()
    assert isinstance(out, SourceIndex)

    out = index.cpu()
    assert isinstance(out, SourceIndex)
    assert out.device == torch.device('cpu')

    out = index.to(torch.int)
    assert out.dtype == torch.int
    if torch_geometric.typing.WITH_PT20:
        assert isinstance(out, SourceIndex)
    else:
        assert not isinstance(out, SourceIndex)

    out = index.to(torch.float)
    assert not isinstance(out, SourceIndex)
    assert out.dtype == torch.float

    out = index.long()
    assert isinstance(out, SourceIndex)
    assert out.dtype == torch.int64

    out = index.int()
    assert out.dtype == torch.int
    if torch_geometric.typing.WITH_PT20:
        assert isinstance(out, SourceIndex)
    else:
        assert not isinstance(out, SourceIndex)


@onlyCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_cpu_cuda(dtype):
    kwargs = dict(dtype=dtype, sparse_size=(3, 4))
    index = SourceIndex([[0], [1], [1], [2]], **kwargs)
    assert index.is_cpu

    out = index.cuda()
    assert isinstance(out, SourceIndex)
    assert out.is_cuda

    out = out.cpu()
    assert isinstance(out, SourceIndex)
    assert out.is_cpu


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_share_memory(dtype, device):
    kwargs = dict(dtype=dtype, device=device, sparse_size=(3, 4))
    index = SourceIndex([[0], [1], [1], [2]], **kwargs)

    out = index.share_memory_()
    assert isinstance(out, SourceIndex)
    assert out.is_shared()
    assert out._data.is_shared()
    assert out.data_ptr() == index.data_ptr()


@onlyCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_pin_memory(dtype):
    index = SourceIndex([[0], [1], [1], [2]], sparse_size=(3, 4), dtype=dtype)
    assert not index.is_pinned()
    out = index.pin_memory()
    assert out.is_pinned()


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_contiguous(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = SourceIndex([[0], [1], [1], [2]], sparse_size=(3, 4), **kwargs)

    assert index.is_contiguous
    out = index.contiguous()
    assert isinstance(out, SourceIndex)
    assert out.is_contiguous


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_sort(dtype, device):
    raise NotImplementedError("This test of SourceIndex has not yet been implemented")


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_sort_stable(dtype, device):
    raise NotImplementedError("This test of SourceIndex has not yet been implemented")


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_cat(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    # todo: handle sorted indices
    index1 = SourceIndex([[0], [1], [1], [2]], sparse_size=(4, 4), **kwargs)
    index2 = SourceIndex([[1], [2], [2], [3]], sparse_size=(4, 4), **kwargs)
    index3 = SourceIndex([[1], [2], [2], [3]], **kwargs)

    out = torch.cat([index1, index2], dim=0)
    assert out.equal(tensor([[0], [1], [1], [2], [1], [2], [2], [3]], device=device))
    assert out.size() == (8, 1)
    assert isinstance(out, SourceIndex)
    assert out.sparse_size == (4, 8)

    assert out._cat_metadata.dim_size == [4, 4]

    out = torch.cat([index1, index2, index3], dim=0)
    assert out.size() == (12, 1)
    assert isinstance(out, SourceIndex)
    assert out.sparse_size == (None, 12)

    out = torch.cat([index1, index2.as_tensor()])
    assert out.size() == (8, 1)
    assert not isinstance(out, SourceIndex)

    inplace = torch.empty((8, 1), dtype=dtype, device=device)
    out = torch.cat([index1, index2], out=inplace)
    assert out.equal(tensor([[0], [1], [1], [2], [1], [2], [2], [3]], device=device))
    assert out.data_ptr() == inplace.data_ptr()
    assert not isinstance(out, SourceIndex)
    assert not isinstance(inplace, SourceIndex)


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_index_select(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = SourceIndex([[0, 1], [1, 1], [1, 2], [2, 2]], sparse_size=(3, 4), **kwargs)

    i = tensor([1, 3], device=device)
    out = index.index_select(0, i)
    assert out.equal(tensor([[1, 1], [2, 2]], device=device))
    assert isinstance(out, SourceIndex)
    assert out.sparse_size == (3, 2)

    inplace = torch.empty((2, 2), dtype=dtype, device=device)
    out = torch.index_select(index, 0, i, out=inplace)
    assert out.equal(tensor([[1, 1], [2, 2]], device=device))
    assert out.data_ptr() == inplace.data_ptr()
    assert not isinstance(out, SourceIndex)
    assert not isinstance(inplace, SourceIndex)


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_getitem(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = SourceIndex([[0, 1], [1, 1], [1, 2], [2, 2]], sparse_size=(3, 4), **kwargs)

    out = index[:, :]
    assert isinstance(out, SourceIndex)
    assert out._data.data_ptr() == index._data.data_ptr()
    assert out.equal(tensor([[0, 1], [1, 1], [1, 2], [2, 2]], device=device))
    assert out.sparse_size == (3, 4)

    out = index[tensor([False, True, False, True], device=device)]
    assert isinstance(out, SourceIndex)
    assert out.equal(tensor([[1, 1], [2, 2]], device=device))
    assert out.sparse_size == (3, 2)

    out = index[tensor([1, 3], device=device)]
    assert isinstance(out, SourceIndex)
    assert out.equal(tensor([[1, 1], [2, 2]], device=device))
    assert out.sparse_size == (3, 2)

    out = index[Index(tensor([1, 3], device=device))]
    assert isinstance(out, SourceIndex)
    assert out.equal(tensor([[1, 1], [2, 2]], device=device))
    assert out.sparse_size == (3, 2)

    out = index[1:3]
    assert isinstance(out, SourceIndex)
    assert out.equal(tensor([[1, 1], [1, 2]], device=device))
    assert out.sparse_size == (3, 2)

    out = index[...]
    assert isinstance(out, SourceIndex)
    assert out._data.data_ptr() == index._data.data_ptr()
    assert out.equal(tensor([[0, 1], [1, 1], [1, 2], [2, 2]], device=device))
    assert out.sparse_size == (3, 4)

    out = index[1:3, ...]
    assert isinstance(out, SourceIndex)
    assert out.equal(tensor([[1, 1], [1, 2]], device=device))
    assert out.sparse_size == (3, 2)

    out = index[0]
    assert not isinstance(out, SourceIndex)
    assert out.equal(tensor([0, 1], device=device))

    tmp = torch.randn(3, device=device)
    out = tmp[index]
    assert not isinstance(out, SourceIndex)
    assert out.equal(tmp[index.as_tensor()])


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_add(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    index = SourceIndex([[0, 1], [1, 1], [1, 2], [2, 2]], sparse_size=(3, 4), **kwargs)

    out = torch.add(index, 2, alpha=2)
    assert isinstance(out, SourceIndex)
    assert out.equal(tensor([[4, 5], [5, 5], [5, 6], [6, 6]], device=device))
    assert out.sparse_size == (7, 4)

    out = index + tensor([2], dtype=dtype, device=device)
    assert isinstance(out, SourceIndex)
    assert out.equal(tensor([[2, 3], [3, 3], [3, 4], [4, 4]], device=device))
    assert out.sparse_size == (5, 4)

    # fixme: not working
    # out = tensor([2], dtype=dtype, device=device) + index
    # assert isinstance(out, SourceIndex)
    # assert out.equal(tensor([[2, 3], [3, 3], [3, 4], [4, 4]], device=device))
    # assert out.sparse_size == (5, 4)

    # out = index.add(index)
    # assert isinstance(out, SourceIndex)
    # assert out.equal(tensor([[0, 2], [2, 2], [2, 4], [4, 4]], device=device))
    # assert out.sparse_size == (6, 4)

    index += 2
    assert isinstance(index, SourceIndex)
    assert out.equal(tensor([[2, 3], [3, 3], [3, 4], [4, 4]], device=device))
    assert out.sparse_size == (5, 4)

    with pytest.raises(RuntimeError, match="can't be cast"):
        index += 2.5


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_sub(dtype, device):
    raise NotImplementedError("This test of SourceIndex has not yet been implemented")


def test_to_list():
    data = torch.tensor([[0, 1], [1, 1], [1, 2], [2, 2]])
    index = SourceIndex(data)
    assert index.tolist() == data.tolist()


def test_numpy():
    data = torch.tensor([[0, 1], [1, 1], [1, 2], [2, 2]])
    index = SourceIndex(data)
    assert np.array_equal(index.numpy(), data.numpy())


@withCUDA
@pytest.mark.parametrize('dtype', DTYPES)
def test_save_and_load(dtype, device, tmp_path):
    kwargs = dict(dtype=dtype, device=device)
    index = SourceIndex([[0, 1], [1, 1], [1, 2], [2, 2]], sparse_size=(3, 4), **kwargs)

    path = osp.join(tmp_path, 'edge_index.pt')
    torch.save(index, path)
    out = fs.torch_load(path)

    assert isinstance(out, SourceIndex)
    assert out.equal(index)
    assert out.sparse_size == (3, 4)


def _collate_fn(indices: List[SourceIndex]) -> List[SourceIndex]:
    return indices


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('num_workers', [0, 2])
@pytest.mark.parametrize('pin_memory', [False, True])
def test_data_loader(dtype, num_workers, pin_memory):
    kwargs = dict(dtype=dtype)
    index = SourceIndex(
        [[0, 1], [1, 1], [1, 2], [2, 2]],
        sparse_size=(3, 4),
        **kwargs
    )

    loader = torch.utils.data.DataLoader(
        [index] * 4,
        batch_size=2,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )

    assert len(loader) == 2
    for batch in loader:
        assert isinstance(batch, list)
        assert len(batch) == 2
        for index in batch:
            assert isinstance(index, SourceIndex)
            assert index.dtype == dtype
            assert index.is_shared() != (num_workers == 0) or pin_memory
            assert index._data.is_shared() != (num_workers == 0) or pin_memory

    # data = Data(edge_index=index, num_nodes=index.num_cols)
    # collated_loader = torch_geometric.data.DataLoader(
    #     [data] * 4,
    #     batch_size=2,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     drop_last=True,
    # )
    #
    # assert len(loader) == 2
    # for batch in collated_loader:
    #     assert isinstance(batch.edge_index, SourceIndex)
    #     assert batch.edge_index.sparse_size == (3, 8)
