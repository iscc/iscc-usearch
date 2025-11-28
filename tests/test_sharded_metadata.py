"""
Test ShardedIndex metadata operations.

Confirms expected behavior for reading index metadata:
- Metadata from valid directory
- Metadata from nonexistent path
- Metadata from file instead of directory
- Metadata from empty directory
"""

import numpy as np

from iscc_usearch.sharded import ShardedIndex


def test_metadata_valid_directory(tmp_path):
    """Test metadata extraction from valid directory."""
    # Create and save
    index1 = ShardedIndex(ndim=64, path=tmp_path)
    index1.add(1, np.random.rand(64).astype(np.float32))
    index1.save()

    meta = ShardedIndex.metadata(tmp_path)

    assert meta is not None
    assert meta["dimensions"] == 64


def test_metadata_nonexistent_path(tmp_path):
    """Test metadata returns None for nonexistent path."""
    result = ShardedIndex.metadata(tmp_path / "nonexistent")

    assert result is None


def test_metadata_file_not_directory(tmp_path):
    """Test metadata returns None for file path."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("test")

    result = ShardedIndex.metadata(file_path)

    assert result is None


def test_metadata_empty_directory(tmp_path):
    """Test metadata returns None for empty directory."""
    result = ShardedIndex.metadata(tmp_path)

    assert result is None
