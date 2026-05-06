"""
Test ShardedIndex properties.

Confirms expected behavior for index properties:
- size/len
- ndim
- dtype
- metric/metric_kind
- connectivity
- expansion_add/expansion_search
- multi
- path
- shard_count
- memory_usage
- serialized_length
- capacity
"""

import numpy as np
from usearch.index import MetricKind

from iscc_usearch.sharded import ShardedIndex


def test_size_property(tmp_path):
    """Test size property."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(list(range(10)), np.random.rand(10, 64).astype(np.float32))

    assert index.size == 10


def test_len_dunder(tmp_path):
    """Test __len__ method."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(list(range(5)), np.random.rand(5, 64).astype(np.float32))

    assert len(index) == 5


def test_ndim_property(tmp_path):
    """Test ndim property."""
    index = ShardedIndex(ndim=128, path=tmp_path)

    assert index.ndim == 128


def test_dtype_property(tmp_path):
    """Test dtype property."""
    index = ShardedIndex(ndim=64, dtype="f32", path=tmp_path)

    # dtype is set by index, not always same as input string
    assert index.dtype is not None


def test_metric_property(tmp_path):
    """Test metric property."""
    index = ShardedIndex(ndim=64, metric=MetricKind.L2sq, path=tmp_path)

    assert index.metric == MetricKind.L2sq


def test_metric_kind_property(tmp_path):
    """Test metric_kind property."""
    index = ShardedIndex(ndim=64, metric=MetricKind.L2sq, path=tmp_path)

    assert index.metric_kind == MetricKind.L2sq


def test_connectivity_property(tmp_path):
    """Test connectivity property."""
    index = ShardedIndex(ndim=64, connectivity=32, path=tmp_path)

    assert index.connectivity == 32


def test_expansion_add_property(tmp_path):
    """Test expansion_add property."""
    index = ShardedIndex(ndim=64, expansion_add=256, path=tmp_path)

    assert index.expansion_add == 256


def test_expansion_add_setter(tmp_path):
    """Test expansion_add setter."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.expansion_add = 512

    assert index.expansion_add == 512


def test_expansion_search_property(tmp_path):
    """Test expansion_search property."""
    index = ShardedIndex(ndim=64, expansion_search=256, path=tmp_path)

    assert index.expansion_search == 256


def test_expansion_search_setter(tmp_path):
    """Test expansion_search setter."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.expansion_search = 512

    assert index.expansion_search == 512


def test_expansion_search_setter_propagates_to_view_shards(tmp_path):
    """Test expansion_search setter propagates to view shards."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=50)
    for i in range(50):
        index.add(i, np.random.rand(64).astype(np.float32))

    # Rotation should have created view shards
    assert len(index._viewed_indexes) > 0

    index.expansion_search = 999
    for shard in index._viewed_indexes:
        assert shard.expansion_search == 999


def test_multi_property(tmp_path):
    """Test multi property."""
    index = ShardedIndex(ndim=64, multi=True, path=tmp_path)

    assert index.multi is True


def test_path_property(tmp_path):
    """Test path property."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    assert index.path == tmp_path


def test_shard_count_property(tmp_path):
    """Test shard_count property."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))
    index.save()

    assert index.shard_count == 1


def test_memory_usage_property(tmp_path):
    """Test memory_usage property."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(list(range(100)), np.random.rand(100, 64).astype(np.float32))

    assert index.memory_usage > 0


def test_memory_usage_includes_view_shards(tmp_path):
    """Test memory_usage includes viewed indexes."""
    # Create index and trigger rotation to get view shards
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)
    for i in range(100):
        index.add(i, np.random.rand(64).astype(np.float32))

    # Should have view shards after rotation
    if index._viewed_indexes:
        memory = index.memory_usage
        # Memory should include both active and view shards
        assert memory > 0
        # Verify view shards contribute to memory
        view_memory = sum(idx.memory_usage for idx in index._viewed_indexes)
        assert view_memory > 0


def test_serialized_length_property(tmp_path):
    """Test serialized_length property."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(list(range(10)), np.random.rand(10, 64).astype(np.float32))

    assert index.serialized_length > 0


def test_capacity_property(tmp_path):
    """Test capacity property."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))

    assert index.capacity > 0


def test_stats(tmp_path):
    """Test stats() returns structured index summary."""
    index = ShardedIndex(ndim=64, path=tmp_path, connectivity=32)
    index.add(list(range(10)), np.random.rand(10, 64).astype(np.float32))

    s = index.stats()

    assert s["total_vectors"] == 10
    assert s["dimensions"] == 64
    assert s["connectivity"] == 32
    assert s["view_shards"] == 0
    assert s["active_shard_vectors"] == 10
    assert s["dirty"] == 10
    assert s["tombstones"] == 0
    assert s["bloom_filter"] is True
    assert s["memory_usage"] > 0
    assert s["path"] == str(tmp_path)
    assert s["read_only"] is False
    assert "metric" in s
    assert "dtype" in s
    assert "shard_size" in s


def test_stats_with_view_shards(tmp_path):
    """Test stats() reflects view shards after rotation."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)
    for i in range(50):
        index.add(i, np.random.rand(64).astype(np.float32))

    s = index.stats()

    assert s["view_shards"] > 0
    assert s["total_vectors"] == 50


def test_repr(tmp_path):
    """Test __repr__ method."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(list(range(5)), np.random.rand(5, 64).astype(np.float32))

    repr_str = repr(index)

    assert "ShardedIndex" in repr_str
    assert "5 vectors" in repr_str
    assert "ndim=64" in repr_str
