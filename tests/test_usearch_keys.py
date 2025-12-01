"""
Confirm the expected behavior of usearch Index.keys property with

- metric=MetricKind.Hamming
- dtype=ScalarKind.B1
- multi=False and multi=True
- Various states: empty, after add, after remove, after load/view
- With enable_key_lookups=True and False
"""

import numpy as np
from numpy.testing import assert_array_equal
from usearch.index import Index, MetricKind, ScalarKind


# Tests for Index.keys property - basic behavior


def test_keys_empty_index_returns_empty():
    """Empty index has empty keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    result = idx.keys
    result_array = np.asarray(result)

    assert len(result) == 0
    assert len(result_array) == 0


def test_keys_single_entry_returns_that_key():
    """Index with one entry returns that key."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(42, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.keys
    result_array = np.asarray(result).astype(np.uint64)

    expected = np.array([42], dtype=np.uint64)
    assert len(result) == 1
    assert_array_equal(result_array, expected)


def test_keys_multiple_entries_returns_all_keys():
    """Index with multiple entries returns all keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(5, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = idx.keys
    result_array = np.asarray(result).astype(np.uint64)

    expected = np.array([1, 2, 5], dtype=np.uint64)
    assert len(result) == 3
    # Keys may not be in insertion order, so sort for comparison
    assert_array_equal(np.sort(result_array), expected)


def test_keys_preserves_insertion_order():
    """Check if keys are returned in insertion order."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    keys_to_add = [100, 5, 999, 1, 50]
    for k in keys_to_add:
        idx.add(k, np.array([k % 256, (k + 1) % 256, (k + 2) % 256, (k + 3) % 256], dtype=np.uint8))

    result = idx.keys
    result_array = np.asarray(result).astype(np.uint64)

    # Check if order matches insertion order
    expected_insertion_order = np.array(keys_to_add, dtype=np.uint64)
    is_insertion_order = np.array_equal(result_array, expected_insertion_order)

    # Document the actual behavior
    print(f"Keys in insertion order: {is_insertion_order}")
    print(f"Inserted: {keys_to_add}")
    print(f"Returned: {result_array.tolist()}")

    # At minimum, all keys should be present
    assert len(result) == len(keys_to_add)
    assert_array_equal(np.sort(result_array), np.sort(expected_insertion_order))


# Tests for Index.keys with large key values


def test_keys_with_large_uint64_values():
    """Keys property handles large uint64 values."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    large_key = 2**63 - 1  # Near max uint64
    idx.add(large_key, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.keys
    result_array = np.asarray(result).astype(np.uint64)

    expected = np.array([large_key], dtype=np.uint64)
    assert len(result) == 1
    assert_array_equal(result_array, expected)


def test_keys_with_zero_key():
    """Keys property handles key value 0."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(0, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.keys
    result_array = np.asarray(result).astype(np.uint64)

    expected = np.array([0], dtype=np.uint64)
    assert len(result) == 1
    assert_array_equal(result_array, expected)


# Tests for Index.keys with multi=True


def test_keys_multi_true_single_vector_per_key():
    """With multi=True, single vector per key returns one key entry."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    result = idx.keys
    result_array = np.asarray(result).astype(np.uint64)

    expected = np.array([1, 2], dtype=np.uint64)
    assert len(result) == 2
    assert_array_equal(np.sort(result_array), expected)


def test_keys_multi_true_multiple_vectors_per_key():
    """With multi=True, multiple vectors per key - check key duplication behavior."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(1, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(1, np.array([1, 2, 3, 4], dtype=np.uint8))
    idx.add(2, np.array([50, 60, 70, 80], dtype=np.uint8))

    result = idx.keys
    result_array = np.asarray(result).astype(np.uint64)

    # Document behavior: does keys return duplicates or unique keys?
    print(f"Index size: {len(idx)}")
    print(f"Keys length: {len(result)}")
    print(f"Keys: {result_array.tolist()}")
    print(f"Unique keys: {np.unique(result_array).tolist()}")

    # The index has 4 vectors total
    assert len(idx) == 4


# Tests for Index.keys after remove


def test_keys_after_remove_excludes_removed_key():
    """After removing a key, it's not in keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    idx.remove(2)

    result = idx.keys
    result_array = np.asarray(result).astype(np.uint64)

    # Key 2 should not be present
    assert 2 not in result_array
    # Keys 1 and 3 should be present
    assert len(result) == 2
    assert_array_equal(np.sort(result_array), np.array([1, 3], dtype=np.uint64))


# Tests for Index.keys after persistence


def test_keys_after_save_and_load(tmp_path):
    """Keys are preserved after save and load."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(5, np.array([1, 2, 3, 4], dtype=np.uint8))

    # Save
    path = tmp_path / "test.usearch"
    idx.save(str(path))

    # Load into new index
    idx2 = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx2.load(str(path))

    result = idx2.keys
    result_array = np.asarray(result).astype(np.uint64)

    expected = np.array([1, 2, 5], dtype=np.uint64)
    assert len(result) == 3
    assert_array_equal(np.sort(result_array), expected)


def test_keys_after_save_and_view(tmp_path):
    """Keys are preserved after save and view."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(5, np.array([1, 2, 3, 4], dtype=np.uint8))

    # Save
    path = tmp_path / "test.usearch"
    idx.save(str(path))

    # View in new index
    idx2 = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx2.view(str(path))

    result = idx2.keys
    result_array = np.asarray(result).astype(np.uint64)

    expected = np.array([1, 2, 5], dtype=np.uint64)
    assert len(result) == 3
    assert_array_equal(np.sort(result_array), expected)


def test_keys_after_restore(tmp_path):
    """Keys are preserved after restore."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(5, np.array([1, 2, 3, 4], dtype=np.uint8))

    # Save
    path = tmp_path / "test.usearch"
    idx.save(str(path))

    # Restore
    idx2 = Index.restore(str(path))

    result = idx2.keys
    result_array = np.asarray(result).astype(np.uint64)

    expected = np.array([1, 2, 5], dtype=np.uint64)
    assert len(result) == 3
    assert_array_equal(np.sort(result_array), expected)


# Tests for Index.keys with enable_key_lookups=False


def test_keys_with_enable_key_lookups_false():
    """Check keys behavior when enable_key_lookups=False."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, enable_key_lookups=False)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    result = idx.keys
    result_array = np.asarray(result).astype(np.uint64)

    # Document the behavior
    print("enable_key_lookups=False")
    print(f"Index size: {len(idx)}")
    print(f"Keys length: {len(result)}")
    print(f"Keys: {result_array.tolist()}")

    # Index still has 2 vectors
    assert len(idx) == 2


def test_keys_after_load_with_enable_key_lookups_false(tmp_path):
    """Check keys behavior after loading with enable_key_lookups=False."""
    # Create index with key lookups enabled
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, enable_key_lookups=True)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    # Save
    path = tmp_path / "test.usearch"
    idx.save(str(path))

    # Load with key lookups disabled
    idx2 = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1, enable_key_lookups=False)
    idx2.load(str(path))

    result = idx2.keys
    result_array = np.asarray(result).astype(np.uint64)

    # Document the behavior
    print("Loaded with enable_key_lookups=False")
    print(f"Index size: {len(idx2)}")
    print(f"Keys length: {len(result)}")
    print(f"Keys: {result_array.tolist()}")


# Tests for Index.keys return type


def test_keys_return_type():
    """Document the exact return type of keys property."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    result = idx.keys

    # Document the type
    print(f"Type of idx.keys: {type(result)}")
    print(f"Type name: {type(result).__name__}")
    print(f"Has __len__: {hasattr(result, '__len__')}")
    print(f"Has __iter__: {hasattr(result, '__iter__')}")
    print(f"Has __array__: {hasattr(result, '__array__')}")

    # Check it can be converted to numpy array
    result_array = np.asarray(result)
    print(f"Array dtype: {result_array.dtype}")
    print(f"Array shape: {result_array.shape}")


def test_keys_iteration():
    """Keys can be iterated."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    result = idx.keys

    # Iterate and collect
    keys_list = list(result)

    assert len(keys_list) == 3
    assert set(keys_list) == {1, 2, 3}


# Tests for batch add with auto-generated keys


def test_keys_with_auto_generated_keys():
    """Keys returns auto-generated keys when None is passed to add."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    # Add with auto-generated keys
    vectors = np.array(
        [
            [178, 204, 60, 240],
            [100, 150, 200, 250],
            [1, 2, 3, 4],
        ],
        dtype=np.uint8,
    )
    generated_keys = idx.add(None, vectors)

    result = idx.keys
    result_array = np.asarray(result).astype(np.uint64)

    # Document behavior
    print(f"Generated keys: {np.asarray(generated_keys).tolist()}")
    print(f"Keys from property: {result_array.tolist()}")

    assert len(result) == 3
    # The keys should match what was returned by add()
    assert_array_equal(np.sort(result_array), np.sort(np.asarray(generated_keys).astype(np.uint64)))


# Tests for lazy/generator behavior of keys


def test_keys_is_lazy_or_materialized():
    """Check if IndexedKeys is a lazy view or materialized copy."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))

    # Get keys reference
    keys1 = idx.keys

    # Add more entries
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    # Get keys again
    keys2 = idx.keys

    # Check if keys1 reflects the new addition (lazy) or not (materialized at access time)
    keys1_array = np.asarray(keys1).astype(np.uint64)
    keys2_array = np.asarray(keys2).astype(np.uint64)

    print(f"keys1 (obtained before adding key 3): {sorted(keys1_array.tolist())}")
    print(f"keys2 (obtained after adding key 3): {sorted(keys2_array.tolist())}")
    print(f"keys1 length: {len(keys1)}, keys2 length: {len(keys2)}")

    # Document: is IndexedKeys a snapshot or live view?
    is_live_view = 3 in keys1_array
    print(f"IndexedKeys is live view: {is_live_view}")


def test_keys_partial_iteration():
    """Check if we can iterate partially without loading all keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    # Add many entries
    for i in range(1000):
        idx.add(i, np.array([i % 256, (i + 1) % 256, (i + 2) % 256, (i + 3) % 256], dtype=np.uint8))

    keys = idx.keys

    # Only iterate first 10 keys
    first_10 = []
    for i, key in enumerate(keys):
        if i >= 10:
            break
        first_10.append(key)

    print(f"First 10 keys from iteration: {first_10}")
    print(f"Total keys available: {len(keys)}")

    assert len(first_10) == 10


def test_keys_iterator_type():
    """Check the type of iterator returned by keys."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    keys = idx.keys
    iterator = iter(keys)

    print(f"Type of iter(keys): {type(iterator)}")
    print(f"Iterator type name: {type(iterator).__name__}")

    # Check if it's a generator
    import types

    is_generator = isinstance(iterator, types.GeneratorType)
    print(f"Is generator: {is_generator}")

    # Get first value
    first = next(iterator)
    print(f"First key: {first}, type: {type(first)}")


def test_keys_multiple_iterations():
    """Check if keys can be iterated multiple times."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    keys = idx.keys

    # First iteration
    first_pass = list(keys)

    # Second iteration
    second_pass = list(keys)

    print(f"First pass: {first_pass}")
    print(f"Second pass: {second_pass}")
    print(f"Same results: {set(first_pass) == set(second_pass)}")

    # If it's a generator, second pass would be empty
    assert len(second_pass) == 3, "Keys should support multiple iterations"


def test_keys_memory_behavior_large_index():
    """Document memory behavior with larger index."""
    import sys

    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    # Add 10000 entries
    num_entries = 10000
    for i in range(num_entries):
        idx.add(i, np.array([i % 256, (i + 1) % 256, (i + 2) % 256, (i + 3) % 256], dtype=np.uint8))

    # Get keys object
    keys = idx.keys

    # Check size of IndexedKeys object itself
    keys_obj_size = sys.getsizeof(keys)
    print(f"Size of IndexedKeys object: {keys_obj_size} bytes")

    # Convert to array and check size
    keys_array = np.asarray(keys)
    array_size = keys_array.nbytes
    print(f"Size of numpy array: {array_size} bytes")
    print(f"Expected size (uint64 * {num_entries}): {num_entries * 8} bytes")

    # The IndexedKeys object size tells us if it holds data or is just a reference
    print(f"IndexedKeys holds data internally: {keys_obj_size > 1000}")


def test_keys_getitem_support():
    """Check if IndexedKeys supports indexing (getitem)."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))
    idx.add(2, np.array([100, 150, 200, 250], dtype=np.uint8))
    idx.add(3, np.array([1, 2, 3, 4], dtype=np.uint8))

    keys = idx.keys

    print(f"Has __getitem__: {hasattr(keys, '__getitem__')}")

    # Try indexing
    try:
        first_key = keys[0]
        print(f"keys[0] = {first_key}")
        print("Indexing supported: True")
    except (TypeError, IndexError) as e:
        print(f"Indexing not supported: {type(e).__name__}: {e}")

    # Try slicing
    try:
        first_two = keys[:2]
        print(f"keys[:2] = {first_two}")
        print("Slicing supported: True")
    except (TypeError, IndexError) as e:
        print(f"Slicing not supported: {type(e).__name__}: {e}")
