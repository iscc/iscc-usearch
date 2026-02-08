"""Tests for ScalableBloomFilter bytes key support."""

import os

from pathlib import Path
from iscc_usearch.bloom import ScalableBloomFilter


def test_add_contains_single_bytes():
    """Test add and contains with single bytes keys."""
    bloom = ScalableBloomFilter()
    key_a = b"\x01" * 16
    key_b = b"\x02" * 16
    key_absent = b"\xff" * 16

    bloom.add(key_a)
    bloom.add(key_b)

    assert bloom.count == 2
    assert bloom.contains(key_a)
    assert bloom.contains(key_b)
    assert not bloom.contains(key_absent)


def test_add_contains_various_lengths():
    """Test bytes keys of various lengths (not limited to 16)."""
    bloom = ScalableBloomFilter()
    keys = [b"", b"\x00", b"short", os.urandom(32), os.urandom(128)]

    for key in keys:
        bloom.add(key)

    assert bloom.count == len(keys)
    for key in keys:
        assert bloom.contains(key)


def test_add_batch_bytes():
    """Test batch add with bytes keys."""
    bloom = ScalableBloomFilter()
    keys = [os.urandom(16) for _ in range(100)]

    bloom.add_batch(keys)

    assert bloom.count == 100
    for key in keys:
        assert bloom.contains(key)


def test_contains_batch_bytes_single_filter():
    """Test contains_batch bytes path with a single filter (fast path)."""
    bloom = ScalableBloomFilter()
    present = [os.urandom(16) for _ in range(5)]
    absent = [os.urandom(16) for _ in range(5)]

    bloom.add_batch(present)
    assert bloom.filter_count == 1

    results = bloom.contains_batch(present + absent)
    for i in range(5):
        assert results[i] is True
    for i in range(5, 10):
        assert results[i] is False


def test_contains_batch_bytes_multiple_filters():
    """Test contains_batch bytes path across multiple filters."""
    bloom = ScalableBloomFilter(initial_capacity=50, fpr=0.01, growth_factor=2.0)
    keys = [os.urandom(16) for _ in range(150)]

    bloom.add_batch(keys)
    assert bloom.filter_count > 1

    absent = [os.urandom(16) for _ in range(5)]
    results = bloom.contains_batch(keys[:5] + absent)
    for i in range(5):
        assert results[i] is True
    for i in range(5, 10):
        assert results[i] is False


def test_add_batch_bytes_empty():
    """Test adding empty bytes batch does nothing."""
    bloom = ScalableBloomFilter()
    bloom.add_batch([])
    assert bloom.count == 0


def test_contains_batch_bytes_empty():
    """Test contains_batch with empty bytes list returns empty."""
    bloom = ScalableBloomFilter()
    bloom.add(b"\x01" * 16)
    assert bloom.contains_batch([]) == []


def test_save_load_roundtrip_bytes(tmp_path: Path):
    """Test save/load round-trip with bytes keys."""
    bloom_path = tmp_path / "bytes.isbf"
    bloom = ScalableBloomFilter(initial_capacity=1000, fpr=0.01)

    keys = [os.urandom(16) for _ in range(50)]
    bloom.add_batch(keys)
    bloom.save(bloom_path)

    loaded = ScalableBloomFilter.load(bloom_path)
    assert loaded.count == bloom.count
    for key in keys:
        assert loaded.contains(key)
    assert not loaded.contains(os.urandom(16))


def test_in_operator_bytes():
    """Test 'in' operator with bytes keys."""
    bloom = ScalableBloomFilter()
    key = b"\xab\xcd" * 8
    bloom.add(key)
    assert key in bloom
    assert b"\x00" * 16 not in bloom


def test_int_keys_unchanged():
    """Verify int keys still work alongside bytes tests (no regression)."""
    bloom = ScalableBloomFilter()
    bloom.add(42)
    bloom.add_batch([100, 200, 300])
    assert bloom.contains(42)
    assert bloom.contains(200)
    assert not bloom.contains(999)
    results = bloom.contains_batch([42, 100, 999])
    assert results == [True, True, False]
