"""
Test ShardedIndex restore operations.

Confirms expected behavior for restoring index from path:
- Restore from valid directory
- Restore in view mode
- Restore from nonexistent path
- Restore from file instead of directory
- Restore from empty directory
"""

import numpy as np

from iscc_usearch.sharded import ShardedIndex


def test_restore_valid_directory(tmp_path):
    """Test restore from valid directory."""
    # Create and save
    index1 = ShardedIndex(ndim=64, path=tmp_path)
    vectors = np.random.rand(10, 64).astype(np.float32)
    index1.add(list(range(10)), vectors)
    index1.save()

    # Restore
    index2 = ShardedIndex.restore(tmp_path)

    assert index2 is not None
    assert len(index2) == 10


def test_restore_view_mode(tmp_path):
    """Test restore in view mode."""
    # Create and save
    index1 = ShardedIndex(ndim=64, path=tmp_path)
    index1.add(1, np.random.rand(64).astype(np.float32))
    index1.save()

    # Restore in view mode
    index2 = ShardedIndex.restore(tmp_path, view=True)

    assert index2 is not None
    assert index2._view_mode is True


def test_restore_nonexistent_path(tmp_path):
    """Test restore returns None for nonexistent path."""
    result = ShardedIndex.restore(tmp_path / "nonexistent")

    assert result is None


def test_restore_file_not_directory(tmp_path):
    """Test restore returns None for file path."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("test")

    result = ShardedIndex.restore(file_path)

    assert result is None


def test_restore_empty_directory(tmp_path):
    """Test restore returns None for empty directory."""
    result = ShardedIndex.restore(tmp_path)

    assert result is None
