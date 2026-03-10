"""Test ShardedIndex search with single-row 2D batch input.

Regression test for GitHub issue #22: AxisError when merging results from
both view shards and active shard with a (1, ndim) shaped query vector.
"""

import numpy as np

from iscc_usearch.sharded import ShardedIndex


def test_search_single_row_batch_with_merge(tmp_path):
    """Search with shape (1, ndim) when both view and active shards have data."""
    ndim = 64
    index = ShardedIndex(ndim=ndim, path=tmp_path, shard_size=4096)

    # Add vectors and save to create view shards
    vectors = np.random.rand(50, ndim).astype(np.float32)
    index.add(list(range(50)), vectors)
    index.save()

    # Add more to active shard so merge path is triggered
    extra = np.random.rand(10, ndim).astype(np.float32)
    index.add(list(range(50, 60)), extra)

    # Verify both view and active shards have data
    assert index._view_shards is not None and len(index._view_shards) > 0
    assert index._active_shard is not None and len(index._active_shard) > 0

    # Single-row 2D input — this triggers the bug
    query = np.random.rand(1, ndim).astype(np.float32)
    result = index.search(query, count=5)

    assert len(result.keys) <= 5
    assert len(result.distances) <= 5
