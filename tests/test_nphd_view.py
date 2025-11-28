"""Tests for NphdIndex.view() method."""

import numpy as np

from iscc_usearch.nphd import NphdIndex


def test_view_from_file(tmp_path):
    """View index from file path."""
    index = NphdIndex(max_dim=128)
    vector = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(42, vector)

    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    viewed = NphdIndex(max_dim=128)
    viewed.view(str(file_path))

    assert 42 in viewed
    np.testing.assert_array_equal(viewed.get(42), vector)


def test_view_from_buffer():
    """View index from buffer."""
    index = NphdIndex(max_dim=128)
    vector = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(42, vector)

    buffer = index.save()

    viewed = NphdIndex(max_dim=128)
    viewed.view(buffer)

    assert 42 in viewed
    np.testing.assert_array_equal(viewed.get(42), vector)


def test_view_restores_max_dim(tmp_path):
    """View correctly restores max_dim from saved ndim."""
    index = NphdIndex(max_dim=192)
    vector = np.random.randint(0, 256, 24, dtype=np.uint8)
    index.add(1, vector)

    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    viewed = NphdIndex(max_dim=128)
    viewed.view(str(file_path))

    assert viewed.max_dim == 192
    assert viewed.max_bytes == 24


def test_view_with_different_max_dims(tmp_path):
    """View works with different max_dim values."""
    for max_dim in [64, 128, 192, 256]:
        max_bytes = max_dim // 8
        index = NphdIndex(max_dim=max_dim)
        vector = np.random.randint(0, 256, max_bytes, dtype=np.uint8)
        index.add(1, vector)

        file_path = tmp_path / f"index_{max_dim}.usearch"
        index.save(str(file_path))

        viewed = NphdIndex(max_dim=64)
        viewed.view(str(file_path))

        assert viewed.max_dim == max_dim
        assert viewed.max_bytes == max_bytes
        np.testing.assert_array_equal(viewed.get(1), vector)


def test_viewed_index_supports_search(tmp_path):
    """Viewed index can perform search operations."""
    index = NphdIndex(max_dim=128)
    vector = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(1, vector)

    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    viewed = NphdIndex(max_dim=128)
    viewed.view(str(file_path))

    matches = viewed.search(vector, count=1)
    assert matches.keys[0] == 1


def test_view_preserves_nphd_metric(tmp_path):
    """Regression test: View preserves custom NPHD metric instead of using standard Hamming."""
    # Create index with two similar vectors (differ by 1 bit)
    index = NphdIndex(max_dim=256)
    vector1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
    vector2 = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=np.uint8)  # Last byte differs by 1 bit
    index.add([100, 200], [vector1, vector2])

    # Search before save - should use NPHD metric
    results_before = index.search(vector1, count=2)
    distances_before = results_before.distances

    # The NPHD distance should be small (around 0.015625 for 1 bit difference in 64-bit vector)
    # If using standard Hamming, it would be 1.0
    assert distances_before[0] == 0.0  # Exact match
    assert distances_before[1] < 0.1  # Small NPHD distance, not 1.0

    # Save and view
    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    viewed = NphdIndex(max_dim=256)
    viewed.view(str(file_path))

    # Search after view - MUST use same NPHD metric
    results_after = viewed.search(vector1, count=2)
    distances_after = results_after.distances

    # Critical: Distances must match (proves NPHD metric is preserved)
    np.testing.assert_array_almost_equal(
        distances_before,
        distances_after,
        decimal=6,
        err_msg="NPHD metric was not preserved after view() - distances changed",
    )
