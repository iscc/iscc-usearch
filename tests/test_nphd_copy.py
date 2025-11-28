"""Tests for NphdIndex.copy() method."""

import numpy as np

from iscc_usearch.nphd import NphdIndex


def test_copy_returns_nphd_index():
    """Copy returns NphdIndex instance."""
    index = NphdIndex(max_dim=128)
    copied = index.copy()
    assert isinstance(copied, NphdIndex)


def test_copy_preserves_max_dim():
    """Copy preserves max_dim and max_bytes."""
    index = NphdIndex(max_dim=192)
    copied = index.copy()
    assert copied.max_dim == 192
    assert copied.max_bytes == 24


def test_copy_preserves_data():
    """Copy preserves all keys and vectors."""
    index = NphdIndex(max_dim=128)
    vector = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(42, vector)

    copied = index.copy()

    assert 42 in copied
    assert len(copied) == 1
    np.testing.assert_array_equal(copied.get(42), vector)


def test_copy_is_independent():
    """Copy is independent - modifying one doesn't affect other."""
    index = NphdIndex(max_dim=128)
    vector1 = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(1, vector1)

    copied = index.copy()

    vector2 = np.random.randint(0, 256, 16, dtype=np.uint8)
    copied.add(2, vector2)

    assert 1 in index
    assert 2 not in index
    assert 1 in copied
    assert 2 in copied


def test_copy_preserves_configuration():
    """Copy preserves connectivity and expansion parameters."""
    index = NphdIndex(max_dim=128, connectivity=32, expansion_add=80, expansion_search=80)
    copied = index.copy()

    assert copied.connectivity == 32
    assert copied.expansion_add == 80
    assert copied.expansion_search == 80
