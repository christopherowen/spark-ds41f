"""Reconstruct a captured pool's native sparse address geometry."""
import torch


def restore_layout(data, max_bytes=8 * 2**30):
    if data.get('schema_version') != 2:
        raise ValueError('snapshot must record native planning and physical layout')
    layout = data['layout']
    pages, stride = int(layout['pool_pages']), int(layout['row_stride_bytes'])
    nbytes = data['pool'].shape[1]
    required = (pages - 1) * stride + nbytes + 255
    if pages <= 0 or stride < nbytes or required > max_bytes:
        raise ValueError(f'captured pool address span {required} exceeds replay budget {max_bytes}')
    backing = torch.empty(required, dtype=torch.uint8, device='cuda')
    offset = (int(layout['alignment_mod_256']) - backing.data_ptr()) % 256
    pool = backing.as_strided((pages, nbytes), (stride, 1), offset)
    ids = layout['physical_ids'].long().cuda()
    # Captured unused bytes are initialized zeros. Other physical pages remain
    # unwritten; native kernels must obey the visible logical table prefix.
    pool.index_copy_(0, ids, data['pool'].cuda())
    table = layout['physical_ids'].long()[data['table'].long()].to(torch.int32).cuda()
    assert pool.stride(0) == stride and pool.data_ptr() % 256 == int(layout['alignment_mod_256'])
    args = [data[k].cuda() for k in ('q', 'scales', 'weights')]
    args.extend([pool, table, data['lengths'].cuda(), data['active'].cuda()])
    return args, required
