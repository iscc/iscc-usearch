"""Tests for NphdIndex.restore() static method."""

import numpy as np

from iscc_usearch.nphd import NphdIndex


def test_restore_from_file(tmp_path):
    """Restore index from file path."""
    index = NphdIndex(max_dim=128)
    vector = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(42, vector)

    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    restored = NphdIndex.restore(str(file_path))

    assert restored is not None
    assert 42 in restored
    np.testing.assert_array_equal(restored.get(42), vector)


def test_restore_from_buffer():
    """Restore index from buffer."""
    index = NphdIndex(max_dim=128)
    vector = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(42, vector)

    buffer = index.save()
    restored = NphdIndex.restore(buffer)

    assert restored is not None
    assert 42 in restored
    np.testing.assert_array_equal(restored.get(42), vector)


def test_restore_with_load(tmp_path):
    """Restore with view=False loads index into memory."""
    index = NphdIndex(max_dim=128)
    vector = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(1, vector)

    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    restored = NphdIndex.restore(str(file_path), view=False)

    assert restored is not None
    assert 1 in restored
    vector2 = np.random.randint(0, 256, 16, dtype=np.uint8)
    restored.add(2, vector2)
    assert 2 in restored


def test_restore_with_view(tmp_path):
    """Restore with view=True memory-maps index."""
    index = NphdIndex(max_dim=128)
    vector = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(1, vector)

    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    restored = NphdIndex.restore(str(file_path), view=True)

    assert restored is not None
    assert 1 in restored
    np.testing.assert_array_equal(restored.get(1), vector)


def test_restore_returns_nphd_index(tmp_path):
    """Restore returns NphdIndex instance."""
    index = NphdIndex(max_dim=128)
    index.add(1, np.random.randint(0, 256, 16, dtype=np.uint8))

    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    restored = NphdIndex.restore(str(file_path))

    assert isinstance(restored, NphdIndex)


def test_restore_invalid_path_returns_none():
    """Restore with invalid path returns None."""
    restored = NphdIndex.restore("nonexistent.usearch")
    assert restored is None


def test_restore_derives_correct_max_dim(tmp_path):
    """Restore correctly derives max_dim from saved metadata."""
    for max_dim in [64, 128, 192, 256]:
        max_bytes = max_dim // 8
        index = NphdIndex(max_dim=max_dim)
        vector = np.random.randint(0, 256, max_bytes, dtype=np.uint8)
        index.add(1, vector)

        file_path = tmp_path / f"index_{max_dim}.usearch"
        index.save(str(file_path))

        restored = NphdIndex.restore(str(file_path))

        assert restored is not None
        assert restored.max_dim == max_dim
        assert restored.max_bytes == max_bytes
        np.testing.assert_array_equal(restored.get(1), vector)


def test_restored_index_supports_operations(tmp_path):
    """Restored index can perform add and search operations."""
    index = NphdIndex(max_dim=128)
    vector1 = np.random.randint(0, 256, 16, dtype=np.uint8)
    index.add(1, vector1)

    file_path = tmp_path / "index.usearch"
    index.save(str(file_path))

    restored = NphdIndex.restore(str(file_path))

    vector2 = np.random.randint(0, 256, 16, dtype=np.uint8)
    restored.add(2, vector2)
    assert 2 in restored

    matches = restored.search(vector1, count=1)
    assert matches.keys[0] == 1
