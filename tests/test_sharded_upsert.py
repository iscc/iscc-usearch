"""
Test ShardedIndex upsert operations.

Confirms expected behavior for insert-or-update by key:
- Upsert new key (acts like add)
- Upsert existing key in active/view shard
- Batch upsert with duplicates (last-write-wins)
- Upsert across rotations
- Error conditions
"""

import numpy as np
import pytest

from iscc_usearch.sharded import ShardedIndex


def test_upsert_new_key(tmp_path):
    """Upsert a new key acts like add."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vec = np.random.rand(64).astype(np.float32)

    index.upsert(1, vec)

    assert index.contains(1)
    result = index.get(1)
    assert result is not None
    assert np.allclose(result, vec, atol=0.01)


def test_upsert_existing_key_active(tmp_path):
    """Upsert replaces vector in active shard."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vec_old = np.ones(64, dtype=np.float32)
    vec_new = np.ones(64, dtype=np.float32) * 2.0

    index.add(1, vec_old)
    index.upsert(1, vec_new)

    result = index.get(1)
    assert result is not None
    assert np.allclose(result, vec_new, atol=0.01)


def test_upsert_existing_key_view(tmp_path):
    """Upsert replaces vector from view shard (tombstone + re-add)."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vec_old = np.ones(64, dtype=np.float32)
    index.add(1, vec_old)
    index.add(2, np.random.rand(64).astype(np.float32))  # force rotation

    vec_new = np.ones(64, dtype=np.float32) * 3.0
    index.upsert(1, vec_new)

    result = index.get(1)
    assert result is not None
    assert np.allclose(result, vec_new, atol=0.01)
    # Tombstone should be cleared by the add
    assert index.tombstone_count == 0


def test_upsert_batch_mixed(tmp_path):
    """Batch upsert with some new and some existing keys."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    index.add(1, np.ones(64, dtype=np.float32))
    index.add(2, np.ones(64, dtype=np.float32) * 2)

    new_vecs = np.array(
        [
            np.ones(64, dtype=np.float32) * 10,  # update key 1
            np.ones(64, dtype=np.float32) * 20,  # update key 2
            np.ones(64, dtype=np.float32) * 30,  # new key 3
        ]
    )

    index.upsert([1, 2, 3], new_vecs)

    for key, expected_val in [(1, 10), (2, 20), (3, 30)]:
        result = index.get(key)
        assert result is not None
        assert np.allclose(result, np.ones(64, dtype=np.float32) * expected_val, atol=0.1)


def test_upsert_batch_duplicates(tmp_path):
    """Batch with duplicate keys — last occurrence wins."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vec_a = np.ones(64, dtype=np.float32) * 1.0
    vec_b = np.ones(64, dtype=np.float32) * 2.0

    index.upsert([1, 1], np.vstack([vec_a, vec_b]))

    result = index.get(1)
    assert result is not None
    # Last occurrence (vec_b) should win
    assert np.allclose(result, vec_b, atol=0.01)


def test_upsert_then_rotate_then_get(tmp_path):
    """Upsert, rotate, verify correct value returned."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)
    vec = np.ones(64, dtype=np.float32) * 5.0
    index.upsert(1, vec)

    # Force rotation
    index.add(2, np.random.rand(64).astype(np.float32))

    result = index.get(1)
    assert result is not None
    assert np.allclose(result, vec, atol=0.01)


def test_upsert_same_key_multiple_rotations(tmp_path):
    """Upsert K, rotate, upsert K, rotate, get K returns latest value."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)

    # First version
    index.upsert(1, np.ones(64, dtype=np.float32) * 1.0)
    index.add(2, np.random.rand(64).astype(np.float32))  # rotation

    # Second version
    index.upsert(1, np.ones(64, dtype=np.float32) * 2.0)
    index.add(3, np.random.rand(64).astype(np.float32))  # rotation

    # Third version
    index.upsert(1, np.ones(64, dtype=np.float32) * 3.0)

    result = index.get(1)
    assert result is not None
    assert np.allclose(result, np.ones(64, dtype=np.float32) * 3.0, atol=0.01)


def test_upsert_multi_raises(tmp_path):
    """Upsert raises ValueError when multi=True."""
    index = ShardedIndex(ndim=64, path=tmp_path, multi=True)

    with pytest.raises(ValueError, match="multi=False"):
        index.upsert(1, np.random.rand(64).astype(np.float32))


def test_upsert_none_key_raises(tmp_path):
    """Upsert raises ValueError when keys is None."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    with pytest.raises(ValueError, match="requires explicit keys"):
        index.upsert(None, np.random.rand(64).astype(np.float32))


def test_upsert_batch_mismatched_lengths(tmp_path):
    """Upsert raises ValueError when keys and vectors counts differ."""
    index = ShardedIndex(ndim=64, path=tmp_path)

    keys = [1, 2, 3]
    vectors = np.random.rand(2, 64).astype(np.float32)
    with pytest.raises(ValueError, match="must match"):
        index.upsert(keys, vectors)


def test_upsert_batch_single_1d_vector(tmp_path):
    """Batch upsert with a single key and 1D vector array."""
    index = ShardedIndex(ndim=64, path=tmp_path)
    vec = np.ones(64, dtype=np.float32) * 7.0

    # Pass list of one key with 1D vector — triggers reshape(1, -1)
    index.upsert([1], vec)

    result = index.get(1)
    assert result is not None
    assert np.allclose(result, vec, atol=0.01)


def test_search_returns_active_version(tmp_path):
    """When same key exists in active+view, search returns active version."""
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)

    # Add key 1 and rotate
    index.add(1, np.ones(64, dtype=np.float32) * 0.5)
    index.add(2, np.random.rand(64).astype(np.float32))

    # Upsert key 1 in active shard
    index.upsert(1, np.ones(64, dtype=np.float32))

    # Search should find key 1 from active shard
    results = index.search(np.ones(64, dtype=np.float32), count=5)
    if 1 in results.keys.tolist():
        idx = results.keys.tolist().index(1)
        # Should be the active version (distance ~ 0)
        assert results.distances[idx] < 0.5


def test_search_suppresses_active_keys_beyond_topk(tmp_path):
    """Active shard key not in active top-k must still suppress stale view copy.

    Scenario: active shard has key K far from query (won't make active top-k),
    view shard has stale K close to query. The stale K must NOT appear because
    active_shard.contains(K) filters it out.
    """
    index = ShardedIndex(ndim=64, path=tmp_path, shard_size=1)

    # Create a distinctive query vector
    query = np.zeros(64, dtype=np.float32)
    query[0] = 1.0

    # Add key 1 with vector CLOSE to query, then rotate to view shard
    close_vec = query.copy()
    close_vec[1] = 0.1  # very similar to query
    index.add(1, close_vec)
    index.add(2, np.random.rand(64).astype(np.float32))  # rotation

    # Upsert key 1 with vector FAR from query — now in active shard
    far_vec = np.ones(64, dtype=np.float32) * -1.0
    index.upsert(1, far_vec)

    # Add many more keys to active shard so key 1 is unlikely to make active top-1
    for i in range(10, 30):
        v = query.copy()
        v[0] += (i - 10) * 0.01
        index.add(i, v)

    # Search with count=1: the stale view copy of key 1 (close to query) must
    # NOT appear — the active shard contains key 1, so the view version is filtered
    results = index.search(query, count=1)
    assert 1 not in results.keys.tolist(), "Stale view copy of key 1 should be suppressed by active_shard.contains()"
