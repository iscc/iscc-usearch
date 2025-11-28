"""Unit tests for NPHD index and vector padding/unpadding functions."""

import numpy as np
import pytest

from iscc_usearch.nphd import NphdIndex, pad_vectors, unpad_vectors


def test_pad_vectors_single():
    """Test padding a single vector."""
    vectors = [np.array([255, 128, 64], dtype=np.uint8)]
    result = pad_vectors(vectors, nbytes=5)

    assert result.shape == (1, 6)  # 1 vector, 5 bytes + 1 length byte
    assert result[0, 0] == 3  # Length prefix
    assert result[0, 1] == 255  # First byte
    assert result[0, 2] == 128  # Second byte
    assert result[0, 3] == 64  # Third byte
    assert result[0, 4] == 0  # Padding
    assert result[0, 5] == 0  # Padding


def test_pad_vectors_multiple():
    """Test padding multiple vectors of different lengths."""
    vectors = [
        np.array([255, 128], dtype=np.uint8),
        np.array([64, 32, 16], dtype=np.uint8),
        np.array([8], dtype=np.uint8),
    ]
    result = pad_vectors(vectors, nbytes=4)

    assert result.shape == (3, 5)  # 3 vectors, 4 bytes + 1 length byte

    # First vector
    assert result[0, 0] == 2
    assert np.array_equal(result[0, 1:3], [255, 128])
    assert np.array_equal(result[0, 3:], [0, 0])

    # Second vector
    assert result[1, 0] == 3
    assert np.array_equal(result[1, 1:4], [64, 32, 16])
    assert result[1, 4] == 0

    # Third vector
    assert result[2, 0] == 1
    assert result[2, 1] == 8
    assert np.array_equal(result[2, 2:], [0, 0, 0])


def test_pad_vectors_truncate_long():
    """Test that vectors longer than nbytes are truncated."""
    vectors = [np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)]
    result = pad_vectors(vectors, nbytes=4)

    assert result.shape == (1, 5)
    assert result[0, 0] == 8  # Original length stored
    assert np.array_equal(result[0, 1:5], [1, 2, 3, 4])  # Only first 4 bytes


def test_pad_vectors_exact_size():
    """Test padding when vector exactly matches nbytes."""
    vectors = [np.array([10, 20, 30], dtype=np.uint8)]
    result = pad_vectors(vectors, nbytes=3)

    assert result.shape == (1, 4)
    assert result[0, 0] == 3
    assert np.array_equal(result[0, 1:], [10, 20, 30])


def test_pad_vectors_2d_input():
    """Test padding with 2D array input (uniform-length vectors)."""
    vectors = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint8)
    result = pad_vectors(vectors, nbytes=4)

    assert result.shape == (3, 5)
    for i in range(3):
        assert result[i, 0] == 2  # All vectors have length 2


def test_unpad_vectors_single():
    """Test unpadding a single vector."""
    padded = np.array([[3, 255, 128, 64, 0, 0]], dtype=np.uint8)
    result = unpad_vectors(padded)

    assert len(result) == 1
    assert len(result[0]) == 3
    assert np.array_equal(result[0], [255, 128, 64])


def test_unpad_vectors_multiple():
    """Test unpadding multiple vectors of different lengths."""
    padded = np.array(
        [
            [2, 255, 128, 0, 0],
            [3, 64, 32, 16, 0],
            [1, 8, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    result = unpad_vectors(padded)

    assert len(result) == 3

    assert len(result[0]) == 2
    assert np.array_equal(result[0], [255, 128])

    assert len(result[1]) == 3
    assert np.array_equal(result[1], [64, 32, 16])

    assert len(result[2]) == 1
    assert np.array_equal(result[2], [8])


def test_unpad_vectors_zero_length():
    """Test unpadding a vector with zero length."""
    padded = np.array([[0, 255, 128, 64, 0]], dtype=np.uint8)
    result = unpad_vectors(padded)

    assert len(result) == 1
    assert len(result[0]) == 0


def test_pad_unpad_roundtrip():
    """Test that pad_vectors and unpad_vectors are inverses."""
    original_vectors = [
        np.array([255, 128, 64], dtype=np.uint8),
        np.array([32, 16], dtype=np.uint8),
        np.array([8, 4, 2, 1], dtype=np.uint8),
    ]

    padded = pad_vectors(original_vectors, nbytes=8)
    recovered = unpad_vectors(padded)

    assert len(recovered) == len(original_vectors)
    for orig, rec in zip(original_vectors, recovered):
        assert np.array_equal(orig, rec)


def test_nphd_index_init_default():
    """Test NphdIndex initialization with default max_dim."""
    index = NphdIndex()

    assert index.max_dim == 256
    assert index.max_bytes == 32
    assert index.ndim == 264  # max_dim + 8 bits for length signal


def test_nphd_index_init_custom_max_dim():
    """Test NphdIndex initialization with custom max_dim."""
    index = NphdIndex(max_dim=128)

    assert index.max_dim == 128
    assert index.max_bytes == 16
    assert index.ndim == 136  # max_dim + 8 bits


def test_nphd_index_init_rejects_ndim():
    """Test that NphdIndex raises assertion error when ndim is provided."""
    with pytest.raises(AssertionError, match="`ndim` is calculated from `max_dim`"):
        NphdIndex(ndim=100)


def test_nphd_index_init_rejects_metric():
    """Test that NphdIndex raises assertion error when metric is provided."""
    with pytest.raises(AssertionError, match="`metric` is set automatically"):
        NphdIndex(metric="hamming")


def test_nphd_index_init_rejects_dtype():
    """Test that NphdIndex raises assertion error when dtype is provided."""
    with pytest.raises(AssertionError, match="`dtype` is set automatically"):
        NphdIndex(dtype="float32")


def test_nphd_index_add_and_search():
    """Test adding vectors and performing search with NphdIndex."""
    index = NphdIndex(max_dim=128)

    # Create test vectors
    vec1 = np.array([255, 128, 64, 32], dtype=np.uint8)
    vec2 = np.array([255, 128, 64, 33], dtype=np.uint8)  # Similar to vec1 (1 bit diff)
    vec3 = np.array([0, 1, 2, 3], dtype=np.uint8)  # Very different

    # Add to index with integer keys (add() handles padding automatically)
    index.add(1, vec1)
    index.add(2, vec2)
    index.add(3, vec3)

    # Search with unpadded query (search() handles padding automatically)
    matches = index.search(vec1, count=1)
    assert matches.keys[0] == 1


def test_nphd_index_batch_add_and_search():
    """Test batch adding vectors and searching with NphdIndex."""
    index = NphdIndex(max_dim=256)

    # Create multiple test vectors
    vectors = [np.array([i, i + 1, i + 2, i + 3], dtype=np.uint8) for i in range(0, 50, 10)]
    keys = list(range(100, 100 + len(vectors)))

    # Batch add to index (add() handles padding automatically)
    index.add(keys, vectors)

    # Verify index size
    assert index.size == len(vectors)

    # Search with unpadded query (search() handles padding automatically)
    matches = index.search(vectors[0], count=1)
    assert matches.keys[0] == keys[0]


def test_nphd_index_variable_length_vectors():
    """Test NphdIndex with variable-length vectors."""
    index = NphdIndex(max_dim=128)

    # Create vectors of different lengths
    short_vec = np.array([255], dtype=np.uint8)
    medium_vec = np.array([255, 128], dtype=np.uint8)
    long_vec = np.array([255, 128, 64], dtype=np.uint8)

    # Add to index (add() handles padding automatically)
    index.add([10, 20, 30], [short_vec, medium_vec, long_vec])

    # Search with unpadded query (search() handles padding automatically)
    matches = index.search(medium_vec, count=3)
    assert 20 in matches.keys  # Should find itself
