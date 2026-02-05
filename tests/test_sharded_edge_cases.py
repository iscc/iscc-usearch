"""Test ShardedIndex edge cases for complete coverage.

Tests edge cases and branches not covered by other test files:
- Operations without bloom filter
- Operations with no active shard
- Property access when active_shard is None
- Empty index scenarios
"""

import numpy as np

from iscc_usearch.sharded import ShardedIndex


def test_contains_batch_without_bloom_filter(tmp_path):
    # type: () -> None
    """Test _contains_batch takes non-bloom path when bloom_filter=False."""
    index = ShardedIndex(ndim=64, path=tmp_path, bloom_filter=False)
    index.add([1, 2, 3], np.random.rand(3, 64).astype(np.float32))

    result = index.contains([1, 2, 999])

    assert result[0]
    assert result[1]
    assert not result[2]


def test_contains_batch_with_no_active_shard(tmp_path):
    # type: () -> None
    """Test _contains_batch with active_shard = None."""
    # Create and save index
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add([1, 2], np.random.rand(2, 64).astype(np.float32))
    index.save()

    # Set active_shard to None to test the branch
    index._active_shard = None

    result = index.contains([1, 999])

    assert result[0]
    assert not result[1]


def test_count_single_with_no_active_shard(tmp_path):
    # type: () -> None
    """Test _count_single with active_shard = None."""
    # Create and save index with multi=True to allow duplicate keys
    index = ShardedIndex(ndim=64, path=tmp_path, multi=True, shard_size=1)
    index.add([1, 1], np.random.rand(2, 64).astype(np.float32))
    index.save()

    # Set active_shard to None
    index._active_shard = None

    count = index.count(1)

    assert count >= 1


def test_count_batch_with_no_active_shard(tmp_path):
    # type: () -> None
    """Test _count_batch with active_shard = None."""
    # Create and save index
    index = ShardedIndex(ndim=64, path=tmp_path, multi=True, shard_size=1)
    index.add([1, 2], np.random.rand(2, 64).astype(np.float32))
    index.save()

    # Set active_shard to None
    index._active_shard = None

    counts = index.count([1, 2])

    assert counts[0] >= 1
    assert counts[1] >= 1


def test_load_existing_empty_directory(tmp_path):
    # type: () -> None
    """Test _load_existing creates active shard when directory is empty."""
    # Create empty directory
    index = ShardedIndex(ndim=64, path=tmp_path)

    # Should create active shard even with empty directory
    assert index._active_shard is not None
    assert len(index) == 0


def test_load_existing_with_bloom_creation(tmp_path):
    # type: () -> None
    """Test _load_existing creates bloom filter when enabled but file missing."""
    # Create index without saving (no bloom file)
    index = ShardedIndex(ndim=64, path=tmp_path, bloom_filter=True)
    index.add(1, np.random.rand(64).astype(np.float32))

    # Force reload to test bloom creation path
    index._load_existing()

    # Should have created bloom filter
    assert index._bloom is not None


def test_property_ndim_without_active_shard(tmp_path):
    # type: () -> None
    """Test ndim property falls back to config when active_shard is None."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index._active_shard = None

    assert index.ndim == 64


def test_property_dtype_without_active_shard(tmp_path):
    # type: () -> None
    """Test dtype property falls back to config when active_shard is None."""
    index = ShardedIndex(ndim=64, path=tmp_path, dtype="f32")
    index._active_shard = None

    # Should return config value or None
    result = index.dtype
    assert result is not None


def test_property_metric_without_active_shard(tmp_path):
    # type: () -> None
    """Test metric property falls back to config when active_shard is None."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index._active_shard = None

    # Should return config value or default
    result = index.metric
    assert result is not None


def test_property_metric_kind_without_active_shard(tmp_path):
    # type: () -> None
    """Test metric_kind property falls back to config when active_shard is None."""
    from usearch.index import MetricKind

    index = ShardedIndex(ndim=64, path=tmp_path, metric=MetricKind.L2sq)
    index._active_shard = None

    result = index.metric_kind
    assert result == MetricKind.L2sq


def test_property_connectivity_without_active_shard(tmp_path):
    # type: () -> None
    """Test connectivity property falls back to config when active_shard is None."""
    index = ShardedIndex(ndim=64, path=tmp_path, connectivity=32)
    index._active_shard = None

    assert index.connectivity == 32


def test_property_expansion_add_without_active_shard(tmp_path):
    # type: () -> None
    """Test expansion_add property falls back to config when active_shard is None."""
    index = ShardedIndex(ndim=64, path=tmp_path, expansion_add=256)
    index._active_shard = None

    assert index.expansion_add == 256


def test_property_expansion_add_setter_without_active_shard(tmp_path):
    # type: () -> None
    """Test expansion_add setter updates config when active_shard is None."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index._active_shard = None

    index.expansion_add = 512

    assert index._config["expansion_add"] == 512


def test_property_expansion_search_without_active_shard(tmp_path):
    # type: () -> None
    """Test expansion_search property falls back to config when active_shard is None."""
    index = ShardedIndex(ndim=64, path=tmp_path, expansion_search=128)
    index._active_shard = None

    assert index.expansion_search == 128


def test_property_expansion_search_setter_without_active_shard(tmp_path):
    # type: () -> None
    """Test expansion_search setter updates config when active_shard is None."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index._active_shard = None

    index.expansion_search = 256

    assert index._config["expansion_search"] == 256


def test_property_multi_without_active_shard(tmp_path):
    # type: () -> None
    """Test multi property falls back to config when active_shard is None."""
    index = ShardedIndex(ndim=64, path=tmp_path, multi=True)
    index._active_shard = None

    assert index.multi is True


def test_property_memory_usage_without_active_shard(tmp_path):
    # type: () -> None
    """Test memory_usage property with active_shard = None."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add([1, 2], np.random.rand(2, 64).astype(np.float32))
    index.save()

    index._active_shard = None

    # Should count only view shards
    usage = index.memory_usage
    assert usage >= 0


def test_property_serialized_length_without_active_shard(tmp_path):
    # type: () -> None
    """Test serialized_length property returns 0 when active_shard is None."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index._active_shard = None

    assert index.serialized_length == 0


def test_property_capacity_without_active_shard(tmp_path):
    # type: () -> None
    """Test capacity property returns 0 when active_shard is None."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index._active_shard = None

    assert index.capacity == 0


def test_property_metric_kind_with_custom_metric(tmp_path):
    # type: () -> None
    """Test metric_kind property with custom CompiledMetric."""
    from iscc_usearch.metrics import create_nphd_metric

    # Create index with custom compiled metric
    custom_metric = create_nphd_metric()
    index = ShardedIndex(ndim=64, path=tmp_path, metric=custom_metric)
    index._active_shard = None

    # Should access metric.kind attribute for custom metrics
    result = index.metric_kind
    assert result is not None


def test_load_existing_empty_shards_with_bloom_enabled(tmp_path):
    # type: () -> None
    """Test _load_existing creates bloom filter when no shards exist but bloom enabled."""
    # Create directory structure
    index_path = tmp_path / "test_index"
    index_path.mkdir()

    # Initialize with empty directory and bloom_filter=True
    index = ShardedIndex(ndim=64, path=index_path, bloom_filter=True)

    # Should create both active shard and bloom filter
    assert index._active_shard is not None
    assert index._bloom is not None




def test_keys_iteration_with_no_active_shard(tmp_path):
    # type: () -> None
    """Test ShardedIndexedKeys iteration with active_shard = None."""
    # Create index with data in view shards
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add([1, 2, 3], np.random.rand(3, 64).astype(np.float32))
    index.save()

    # Set active_shard to None to test the exit branch
    index._active_shard = None

    # Should still iterate over view shards
    keys_list = list(index.keys)
    assert len(keys_list) >= 3


def test_vectors_iteration_with_no_active_shard(tmp_path):
    # type: () -> None
    """Test ShardedIndexedVectors iteration with active_shard = None."""
    # Create index with data in view shards
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    index.add([1, 2, 3], np.random.rand(3, 64).astype(np.float32))
    index.save()

    # Set active_shard to None to test the exit branch
    index._active_shard = None

    # Should still iterate over view shards
    vectors_list = list(index.vectors)
    assert len(vectors_list) >= 3


def test_get_single_bloom_rejects_key(tmp_path):
    # type: () -> None
    """Test _get_single early return when bloom filter rejects key."""
    index = ShardedIndex(ndim=64, path=tmp_path, bloom_filter=True)
    index.add([1, 2, 3], np.random.rand(3, 64).astype(np.float32))

    # Get a key that definitely doesn't exist
    # Bloom filter should reject it quickly
    result = index.get(999999)

    assert result is None


def test_get_single_from_active_shard(tmp_path):
    # type: () -> None
    """Test _get_single finds key in active shard."""
    index = ShardedIndex(ndim=64, path=tmp_path, bloom_filter=True)
    vec = np.random.rand(64).astype(np.float32)
    index.add(42, vec)

    # Key should be in active shard
    result = index.get(42)

    assert result is not None
    assert np.allclose(result, vec, atol=0.01)


def test_contains_single_bloom_rejects_key(tmp_path):
    # type: () -> None
    """Test _contains_single early return when bloom filter rejects key."""
    index = ShardedIndex(ndim=64, path=tmp_path, bloom_filter=True)
    index.add([1, 2, 3], np.random.rand(3, 64).astype(np.float32))

    # Check a key that definitely doesn't exist
    # Bloom filter should reject it quickly
    result = index.contains(999999)

    assert result is False


def test_contains_single_from_active_shard(tmp_path):
    # type: () -> None
    """Test _contains_single finds key in active shard."""
    index = ShardedIndex(ndim=64, path=tmp_path, bloom_filter=True)
    index.add(42, np.random.rand(64).astype(np.float32))

    # Key should be in active shard
    result = index.contains(42)

    assert result is True


def test_get_single_with_no_active_shard(tmp_path):
    # type: () -> None
    """Test _get_single with active_shard = None (branch 426->431)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1, bloom_filter=False)
    index.add([1, 2, 3], np.random.rand(3, 64).astype(np.float32))
    index.save()

    # Set active_shard to None to test branch 426->431
    index._active_shard = None

    # Should find key in view shards
    result = index.get(1)

    assert result is not None


def test_contains_single_with_no_active_shard(tmp_path):
    # type: () -> None
    """Test _contains_single with active_shard = None (branch 519->524)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1, bloom_filter=False)
    index.add([1, 2, 3], np.random.rand(3, 64).astype(np.float32))
    index.save()

    # Set active_shard to None to test branch 519->524
    index._active_shard = None

    # Should find key in view shards
    result = index.contains(1)

    assert result is True


def test_get_single_key_not_in_active_shard_but_in_view(tmp_path):
    # type: () -> None
    """Test _get_single when key not in active shard but in view shard."""
    path = tmp_path / "idx"
    # Create index with tiny shard_size to force rotation, bloom disabled
    index = ShardedIndex(ndim=64, path=path, shard_size=1, bloom_filter=False)

    # Add first key - goes to active shard, then rotates to view shard
    index.add(1, np.random.rand(64).astype(np.float32))
    # Now: view_shards has key 1, active_shard is new and empty

    # Add another key - goes to new active shard
    index.add(2, np.random.rand(64).astype(np.float32))
    # Now: view_shards has keys 1,2 (after rotation), active_shard has key 2 or is empty

    # Verify we have view shards
    assert len(index._viewed_indexes) >= 1

    # Get key 1 which is in view shard, NOT in current active shard
    # This tests branch 426->431: active_shard exists but doesn't contain key
    result = index.get(1)

    assert result is not None


def test_contains_single_key_not_in_active_shard_but_in_view(tmp_path):
    # type: () -> None
    """Test _contains_single when key not in active shard but in view shard."""
    path = tmp_path / "idx"
    # Create index with tiny shard_size to force rotation, bloom disabled
    index = ShardedIndex(ndim=64, path=path, shard_size=1, bloom_filter=False)

    # Add first key - goes to active shard, then rotates to view shard
    index.add(1, np.random.rand(64).astype(np.float32))
    # Now: view_shards has key 1, active_shard is new and empty

    # Add another key - goes to new active shard
    index.add(2, np.random.rand(64).astype(np.float32))
    # Now: view_shards has keys 1,2 (after rotation), active_shard has key 2 or is empty

    # Verify we have view shards
    assert len(index._viewed_indexes) >= 1

    # Check key 1 which is in view shard, NOT in current active shard
    # This tests branch 519->524: active_shard exists but doesn't contain key
    result = index.contains(1)

    assert result is True


def test_get_batch_without_bloom_filter(tmp_path):
    # type: () -> None
    """Test _get_batch without bloom filter."""
    index = ShardedIndex(ndim=64, path=tmp_path, bloom_filter=False)
    vec1 = np.random.rand(64).astype(np.float32)
    vec2 = np.random.rand(64).astype(np.float32)
    vectors = np.vstack([vec1, vec2])
    index.add([1, 2], vectors)

    # This tests the branch: bloom filter disabled (448->455)
    results = index.get([1, 2, 999])

    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is not None
    assert results[2] is None


def test_get_batch_with_active_shard_none_and_bloom(tmp_path):
    # type: () -> None
    """Test _get_batch with active_shard = None but bloom enabled."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1, bloom_filter=True)
    index.add([1, 2, 3], np.random.rand(3, 64).astype(np.float32))
    index.save()

    # Set active_shard to None
    index._active_shard = None

    # This tests the branch: active_shard is None (480->484)
    results = index.get([1, 2])

    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is not None


def test_get_single_from_view_shard_after_checking_multiple_shards(tmp_path):
    # type: () -> None
    """Test _get_single checks multiple view shards."""
    path = tmp_path / "idx"
    index = ShardedIndex(ndim=64, path=path, shard_size=50, bloom_filter=True)

    # Add enough to create multiple view shards
    for i in range(100):
        index.add(i, np.random.rand(64).astype(np.float32))

    index.save()

    # Reload to have multiple view shards
    index2 = ShardedIndex(ndim=64, path=path, bloom_filter=True)
    assert len(index2._viewed_indexes) >= 2

    # Get a key from a later view shard
    # This tests the branch: loop through view shards (432->431)
    result = index2.get(50)

    assert result is not None


def test_contains_single_from_view_shard_after_checking_multiple_shards(tmp_path):
    # type: () -> None
    """Test _contains_single checks multiple view shards."""
    path = tmp_path / "idx"
    index = ShardedIndex(ndim=64, path=path, shard_size=50, bloom_filter=True)

    # Add enough to create multiple view shards
    for i in range(100):
        index.add(i, np.random.rand(64).astype(np.float32))

    index.save()

    # Reload to have multiple view shards
    index2 = ShardedIndex(ndim=64, path=path, bloom_filter=True)
    assert len(index2._viewed_indexes) >= 2

    # Check a key from a later view shard
    # This tests the branch: loop through view shards (525->524)
    result = index2.contains(50)

    assert result is True
