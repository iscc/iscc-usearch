"""Tests for ScalableBloomFilter."""

import pytest
from pathlib import Path
from iscc_usearch.bloom import ScalableBloomFilter


def test_bloom_init():
    """Test bloom filter initialization."""
    bloom = ScalableBloomFilter()
    assert bloom.count == 0
    assert bloom.filter_count == 1
    assert bloom.current_capacity > 0


def test_bloom_init_custom_params():
    """Test bloom filter with custom parameters."""
    bloom = ScalableBloomFilter(initial_capacity=1000, fpr=0.001, growth_factor=4.0)
    assert bloom.count == 0
    assert bloom._initial_capacity == 1000
    assert bloom._fpr == 0.001
    assert bloom._growth_factor == 4.0


def test_bloom_add_single():
    """Test adding a single key."""
    bloom = ScalableBloomFilter()
    bloom.add(42)
    assert bloom.count == 1
    assert bloom.contains(42)
    assert not bloom.contains(999)


def test_bloom_add_batch():
    """Test adding multiple keys."""
    bloom = ScalableBloomFilter()
    keys = [1, 2, 3, 4, 5]
    bloom.add_batch(keys)
    assert bloom.count == 5
    for key in keys:
        assert bloom.contains(key)


def test_bloom_contains_batch():
    """Test checking multiple keys."""
    bloom = ScalableBloomFilter()
    bloom.add_batch([1, 3, 5])
    results = bloom.contains_batch([1, 2, 3, 4, 5])
    assert results[0] is True  # 1 exists
    assert results[1] is False  # 2 doesn't exist
    assert results[2] is True  # 3 exists
    assert results[3] is False  # 4 doesn't exist
    assert results[4] is True  # 5 exists


def test_bloom_in_operator():
    """Test 'in' operator support."""
    bloom = ScalableBloomFilter()
    bloom.add(100)
    assert 100 in bloom
    assert 200 not in bloom


def test_bloom_len():
    """Test __len__ method."""
    bloom = ScalableBloomFilter()
    assert len(bloom) == 0
    bloom.add(1)
    assert len(bloom) == 1
    bloom.add_batch([2, 3, 4])
    assert len(bloom) == 4


def test_bloom_clear():
    """Test clearing the bloom filter."""
    bloom = ScalableBloomFilter()
    bloom.add_batch([1, 2, 3, 4, 5])
    assert bloom.count == 5
    bloom.clear()
    assert bloom.count == 0
    assert bloom.filter_count == 1
    assert not bloom.contains(1)


def test_bloom_growth():
    """Test that bloom filter grows when capacity is reached."""
    # Small initial capacity to trigger growth
    bloom = ScalableBloomFilter(initial_capacity=100, fpr=0.01, growth_factor=2.0)
    initial_filter_count = bloom.filter_count

    # Add more elements than initial capacity
    for i in range(250):
        bloom.add(i)

    # Should have grown
    assert bloom.filter_count > initial_filter_count
    assert bloom.count == 250

    # All elements should still be found
    for i in range(250):
        assert bloom.contains(i), f"Key {i} not found after growth"


def test_bloom_save_load(tmp_path: Path):
    """Test saving and loading bloom filter."""
    bloom_path = tmp_path / "test.isbf"

    # Create and populate bloom filter
    bloom = ScalableBloomFilter(initial_capacity=1000, fpr=0.01)
    bloom.add_batch([10, 20, 30, 40, 50])

    # Save
    bloom.save(bloom_path)
    assert bloom_path.exists()

    # Load
    loaded = ScalableBloomFilter.load(bloom_path)
    assert loaded.count == bloom.count
    assert loaded._fpr == bloom._fpr
    assert loaded._initial_capacity == bloom._initial_capacity
    assert loaded._growth_factor == bloom._growth_factor

    # Verify contents
    assert loaded.contains(10)
    assert loaded.contains(30)
    assert loaded.contains(50)
    assert not loaded.contains(99)


def test_bloom_save_load_multiple_filters(tmp_path: Path):
    """Test saving/loading bloom filter with multiple internal filters."""
    bloom_path = tmp_path / "multi.isbf"

    # Create bloom with small capacity to force multiple filters
    bloom = ScalableBloomFilter(initial_capacity=100, fpr=0.01, growth_factor=2.0)

    # Add enough elements to create multiple filters
    for i in range(500):
        bloom.add(i)

    assert bloom.filter_count > 1, "Should have multiple filters"

    # Save and reload
    bloom.save(bloom_path)
    loaded = ScalableBloomFilter.load(bloom_path)

    # Verify structure
    assert loaded.filter_count == bloom.filter_count
    assert loaded.count == bloom.count

    # Verify all elements still accessible
    for i in range(500):
        assert loaded.contains(i), f"Key {i} not found after load"


def test_bloom_load_invalid_magic(tmp_path: Path):
    """Test loading file with invalid magic raises error."""
    bad_path = tmp_path / "bad.isbf"
    bad_path.write_bytes(b"XXXX" + b"\x00" * 100)

    with pytest.raises(ValueError, match="Invalid bloom filter file"):
        ScalableBloomFilter.load(bad_path)


def test_bloom_load_invalid_version(tmp_path: Path):
    """Test loading file with unsupported version raises error."""
    bad_path = tmp_path / "badver.isbf"
    # Write valid magic but bad version
    bad_path.write_bytes(b"ISBF" + b"\xff" + b"\x00" * 100)

    with pytest.raises(ValueError, match="Unsupported bloom filter version"):
        ScalableBloomFilter.load(bad_path)


def test_bloom_repr():
    """Test string representation."""
    bloom = ScalableBloomFilter(initial_capacity=1000, fpr=0.01)
    bloom.add_batch([1, 2, 3])
    repr_str = repr(bloom)
    assert "ScalableBloomFilter" in repr_str
    assert "count=3" in repr_str
    assert "filters=1" in repr_str


def test_bloom_false_positive_rate():
    """Test that false positive rate is within expected bounds."""
    bloom = ScalableBloomFilter(initial_capacity=10000, fpr=0.01)

    # Add 10000 elements
    for i in range(10000):
        bloom.add(i)

    # Test 10000 elements that definitely don't exist
    false_positives = 0
    for i in range(10000, 20000):
        if bloom.contains(i):
            false_positives += 1

    # FPR should be around 1%, allow some margin (up to 3%)
    fpr = false_positives / 10000
    assert fpr < 0.03, f"False positive rate {fpr:.2%} too high (expected ~1%)"


def test_bloom_no_false_negatives():
    """Test that there are no false negatives."""
    bloom = ScalableBloomFilter()

    # Add elements
    keys = list(range(1000))
    bloom.add_batch(keys)

    # All added elements must be found (no false negatives)
    for key in keys:
        assert bloom.contains(key), f"False negative for key {key}"


def test_bloom_add_batch_empty():
    """Test adding empty batch does nothing."""
    bloom = ScalableBloomFilter()
    initial_count = bloom.count
    bloom.add_batch([])
    assert bloom.count == initial_count


def test_bloom_contains_batch_empty():
    """Test contains_batch with empty keys returns empty list."""
    bloom = ScalableBloomFilter()
    bloom.add(1)
    assert bloom.contains_batch([]) == []


def test_bloom_contains_batch_multiple_filters():
    """Test contains_batch works across multiple internal filters."""
    bloom = ScalableBloomFilter(initial_capacity=100, fpr=0.01, growth_factor=2.0)

    # Add enough to create multiple filters
    for i in range(250):
        bloom.add(i)
    assert bloom.filter_count > 1

    # Batch check keys from different filters and non-existent keys
    results = bloom.contains_batch([0, 50, 150, 249, 9999])
    assert results[0] is True  # in first filter
    assert results[1] is True  # in first filter
    assert results[2] is True  # in second filter
    assert results[3] is True  # in latest filter
    assert results[4] is False  # never added


def test_bloom_add_batch_when_filter_exactly_full():
    """Test adding batch when current filter is exactly at capacity."""
    # Create bloom with tiny capacity to force exact-fill scenario
    bloom = ScalableBloomFilter(initial_capacity=10, fpr=0.01, growth_factor=2.0)

    # Fill first filter exactly to capacity
    for i in range(10):
        bloom.add(i)

    # Now add more via batch - should trigger growth
    bloom.add_batch([100, 101, 102])

    # Should have grown to accommodate
    assert bloom.filter_count >= 2
    assert bloom.count == 13

    # All keys should be present
    for i in range(10):
        assert bloom.contains(i)
    for i in [100, 101, 102]:
        assert bloom.contains(i)
