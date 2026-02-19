"""
Test ShardedIndex unsupported operations.

Confirms that unsupported operations raise NotImplementedError:
- rename
- join
- cluster
- pairwise_distance
- copy
- clear
"""

import pytest

from iscc_usearch.sharded import ShardedIndex


def test_rename_not_supported(tmp_path):
    """Test rename raises NotImplementedError."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    with pytest.raises(NotImplementedError, match="not supported"):
        index.rename(1, 2)


def test_join_not_supported(tmp_path):
    """Test join raises NotImplementedError."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    with pytest.raises(NotImplementedError, match="not supported"):
        index.join()


def test_cluster_not_supported(tmp_path):
    """Test cluster raises NotImplementedError."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    with pytest.raises(NotImplementedError, match="not supported"):
        index.cluster()


def test_pairwise_distance_not_supported(tmp_path):
    """Test pairwise_distance raises NotImplementedError."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    with pytest.raises(NotImplementedError, match="not supported"):
        index.pairwise_distance()


def test_copy_not_supported(tmp_path):
    """Test copy raises NotImplementedError."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    with pytest.raises(NotImplementedError, match="not supported"):
        index.copy()


def test_clear_not_supported(tmp_path):
    """Test clear raises NotImplementedError."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    with pytest.raises(NotImplementedError, match="not supported"):
        index.clear()
