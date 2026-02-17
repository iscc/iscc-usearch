"""
Test ShardedIndex save operations.

Confirms expected behavior for persisting index to disk:
- Save and reload roundtrip
- Empty index handling
- No duplicate shards on repeated saves
"""

import numpy as np
import pytest

from iscc_usearch.sharded import ShardedIndex


def test_save_and_load(tmp_path):
    """Test save and load roundtrip."""
    # Create and save
    index1 = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index1.add(list(range(10)), vectors)
    index1.save()

    # Load
    index2 = ShardedIndex(ndim=64, path=tmp_path)

    assert len(index2) == 10


def test_save_rejects_path_argument(tmp_path):
    """Test save() raises TypeError when path_or_buffer is provided."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.random.rand(64).astype(np.float32))

    with pytest.raises(TypeError, match="does not accept a path argument"):
        index.save("/some/path")


def test_save_empty_index(tmp_path):
    """Test save does nothing for empty index."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.save()

    assert index.shard_count == 0


def test_save_no_duplicate_shards(tmp_path):
    """Test save doesn't create duplicate shards (P1 fix)."""
    # Create and save
    index1 = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index1.add(list(range(10)), vectors)
    index1.save()

    # Reopen and add more
    index2 = ShardedIndex(ndim=64, path=tmp_path)
    index2.add(100, np.random.rand(64).astype(np.float32))
    index2.save()

    # Should still be 1 shard, not 2
    assert index2.shard_count == 1


def test_save_no_temp_files_after_save(tmp_path):
    """After save(), no .tmp files should remain."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index.add(list(range(10)), vectors)
    index.save()

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Stale temp files found: {tmp_files}"


def test_save_no_temp_files_after_rotation(tmp_path):
    """After shard rotation, no .tmp files should remain."""
    index = ShardedIndex(ndim=32, path=tmp_path, shard_size=500)

    for i in range(100):
        vector = np.random.rand(32).astype(np.float32)
        index.add(i, vector)

    assert index.shard_count >= 2
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Stale temp files found: {tmp_files}"


def test_stale_tmp_cleanup_on_load(tmp_path):
    """Stale .tmp files from interrupted saves are cleaned up on load."""
    # Create a valid index
    index = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index.add(list(range(10)), vectors)
    index.save()

    # Simulate stale temp files from a crash
    (tmp_path / "shard_000.usearch.tmp").write_bytes(b"stale")
    (tmp_path / "bloom.isbf.tmp").write_bytes(b"stale")

    # Reopen index - should clean up stale files
    index2 = ShardedIndex(ndim=64, path=tmp_path)

    assert not (tmp_path / "shard_000.usearch.tmp").exists()
    assert not (tmp_path / "bloom.isbf.tmp").exists()
    assert len(index2) == 10
