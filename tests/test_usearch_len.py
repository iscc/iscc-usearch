"""
Confirm the expected behavior of usearch Index.__len__() with

- metric=MetricKind.Hamming
- dtype=ScalarKind.B1
- multi=False and multi=True
- Empty index, after adds, after removes
"""

import numpy as np
from usearch.index import Index, MetricKind, ScalarKind


# Tests for Index.__len__() basic behavior


def test_len_empty_index_returns_zero():
    """Empty index has length zero."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)

    result = len(idx)

    expected = 0
    assert result == expected
    assert isinstance(result, int)


def test_len_after_single_add_returns_one():
    """After adding one vector, length is one."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = len(idx)

    expected = 1
    assert result == expected


def test_len_after_batch_add_returns_batch_size():
    """After adding multiple vectors, length equals count of vectors."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    vectors = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
            [1, 2, 3, 4],
        ],
        dtype=np.uint8,
    )
    idx.add([1, 2, 3], vectors)

    result = len(idx)

    expected = 3
    assert result == expected


def test_len_after_remove_decreases():
    """After removing a vector, length decreases by one."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.remove(1)

    result = len(idx)

    expected = 1
    assert result == expected


def test_len_after_removing_all_returns_zero():
    """After removing all vectors, length returns to zero."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.remove(1)
    idx.remove(2)

    result = len(idx)

    expected = 0
    assert result == expected


# Tests for Index.__len__() with multi=True


def test_len_with_multi_true_counts_vectors_not_keys():
    """With multi=True, len() returns total vector count, not unique key count."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    # Add 3 vectors under the same key
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = len(idx)

    # Length is 3 (vectors), not 1 (unique keys)
    expected = 3
    assert result == expected


def test_len_with_multi_true_multiple_keys():
    """With multi=True and multiple keys, len() counts all vectors."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    # Key 1: 2 vectors
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))
    # Key 2: 1 vector
    idx.add(2, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = len(idx)

    expected = 3
    assert result == expected


# Tests for len() vs size property equivalence


def test_len_matches_size_property():
    """len(index) equals index.size property."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    len_result = len(idx)
    size_result = idx.size

    assert len_result == size_result
    assert len_result == 2


def test_len_matches_size_property_multi_true():
    """len(index) equals index.size property with multi=True."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))

    len_result = len(idx)
    size_result = idx.size

    assert len_result == size_result
    assert len_result == 2
