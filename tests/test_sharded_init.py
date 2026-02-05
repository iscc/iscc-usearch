"""
Test ShardedIndex initialization.

Confirms expected behavior when creating ShardedIndex with various parameter combinations:
- ndim (dimensionality)
- path (directory for shards)
- configuration options (dtype, connectivity, expansion, multi, etc.)
- loading existing shards
"""

import numpy as np

from iscc_usearch.sharded import ShardedIndex, DEFAULT_SHARD_SIZE


def test_init_creates_directory(tmp_path):
    """Test that init creates the directory if it doesn't exist."""
    new_dir = tmp_path / "subdir" / "index"
    index = ShardedIndex(ndim=64, path=new_dir)

    assert new_dir.exists()
    assert index._active_shard is not None
    assert len(index) == 0


def test_init_empty_directory(tmp_path):
    """Test init with empty directory creates active shard."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    assert index._active_shard is not None
    assert index._view_shards is None
    assert index.shard_count == 0


def test_init_loads_existing_shards(tmp_path):
    """Test init loads existing shards."""
    # Create and save an index
    index1 = ShardedIndex(ndim=64, path=tmp_path, shard_size=1000)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index1.add(list(range(10)), vectors)
    index1.save()

    # Reopen - should load existing shards
    index2 = ShardedIndex(ndim=64, path=tmp_path)

    assert len(index2) == 10
    assert index2.shard_count == 1


def test_init_with_all_config_options(tmp_path):
    """Test init with all configuration options."""
    index = ShardedIndex(
        ndim=128,
        dtype="f32",
        connectivity=32,
        expansion_add=256,
        expansion_search=128,
        multi=True,
        path=tmp_path,
        shard_size=1024,
    )

    assert index.ndim == 128
    assert index.connectivity == 32
    assert index.expansion_add == 256
    assert index.expansion_search == 128
    assert index.multi is True


def test_default_shard_size():
    """Test DEFAULT_SHARD_SIZE constant."""
    assert DEFAULT_SHARD_SIZE == 1024 * 1024 * 1024  # 1GB


def test_init_without_ndim_raises_when_no_shards(tmp_path):
    """Test that init without ndim raises error when no existing shards."""
    import pytest

    with pytest.raises(ValueError, match="ndim is required"):
        ShardedIndex(path=tmp_path)


def test_init_without_ndim_autodetects_from_existing_shards(tmp_path):
    """Test that init without ndim auto-detects from existing shards."""
    # Create and save an index with known ndim
    index1 = ShardedIndex(ndim=128, path=tmp_path)
    vectors = np.random.rand(5, 128).astype(np.float32)
    index1.add(list(range(5)), vectors)
    index1.save()

    # Reopen without specifying ndim - should auto-detect
    index2 = ShardedIndex(path=tmp_path)

    assert index2.ndim == 128
    assert len(index2) == 5
