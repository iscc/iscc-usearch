"""
Test ShardedIndex unsupported operations.

Confirms that append-only and unsupported operations raise NotImplementedError:
- remove
- __delitem__
- rename
- join
- cluster
- pairwise_distance
- copy
- clear
- reset
"""

import pytest

from iscc_usearch.sharded import ShardedIndex


def test_remove_not_supported(tmp_path):
    """Test remove raises NotImplementedError."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    with pytest.raises(NotImplementedError, match="append-only"):
        index.remove(1)


def test_delitem_not_supported(tmp_path):
    """Test __delitem__ raises NotImplementedError."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    with pytest.raises(NotImplementedError, match="append-only"):
        del index[1]


def test_rename_not_supported(tmp_path):
    """Test rename raises NotImplementedError."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    with pytest.raises(NotImplementedError, match="append-only"):
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


def test_reset_not_supported(tmp_path):
    """Test reset raises NotImplementedError."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    with pytest.raises(NotImplementedError, match="not supported"):
        index.reset()
