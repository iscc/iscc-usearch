"""Tests for the dirty write counter across all index types.

Validates that the `dirty` property correctly tracks unsaved key mutations
and resets after save/load/view/reset operations.
"""

import numpy as np

from iscc_usearch import NphdIndex, ShardedIndex, ShardedIndex128, ShardedNphdIndex, ShardedNphdIndex128


# === NphdIndex dirty counter ===


class TestNphdIndexDirty:
    """Tests for NphdIndex.dirty property."""

    def test_dirty_starts_at_zero(self):
        idx = NphdIndex(max_dim=64)
        assert idx.dirty == 0

    def test_dirty_increments_on_single_add(self):
        idx = NphdIndex(max_dim=64)
        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec)
        assert idx.dirty == 1

    def test_dirty_increments_on_batch_add(self):
        idx = NphdIndex(max_dim=64)
        vecs = np.random.randint(0, 256, size=(5, 8), dtype=np.uint8)
        idx.add(list(range(5)), vecs)
        assert idx.dirty == 5

    def test_dirty_increments_on_remove(self):
        idx = NphdIndex(max_dim=64)
        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec)
        idx.remove(1)
        assert idx.dirty == 2  # 1 add + 1 remove

    def test_dirty_accumulates_across_operations(self):
        idx = NphdIndex(max_dim=64)
        for i in range(10):
            vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
            idx.add(i, vec)
        assert idx.dirty == 10

        idx.remove(0)
        assert idx.dirty == 11

    def test_dirty_resets_on_save(self, tmp_path):
        idx = NphdIndex(max_dim=64)
        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec)
        assert idx.dirty == 1

        idx.save(str(tmp_path / "test.usearch"))
        assert idx.dirty == 0

    def test_dirty_resets_on_load(self, tmp_path):
        # Create and save an index
        idx = NphdIndex(max_dim=64)
        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec)
        path = str(tmp_path / "test.usearch")
        idx.save(path)

        # Create a new index, mutate it, then load
        idx2 = NphdIndex(max_dim=64)
        idx2.add(2, vec)
        assert idx2.dirty == 1
        idx2.load(path)
        assert idx2.dirty == 0

    def test_dirty_resets_on_view(self, tmp_path):
        # Create and save an index
        idx = NphdIndex(max_dim=64)
        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec)
        path = str(tmp_path / "test.usearch")
        idx.save(path)

        # Create a new index, mutate it, then view
        idx2 = NphdIndex(max_dim=64)
        idx2.add(2, vec)
        assert idx2.dirty == 1
        idx2.view(path)
        assert idx2.dirty == 0

    def test_dirty_resets_on_reset(self):
        idx = NphdIndex(max_dim=64)
        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec)
        assert idx.dirty == 1

        idx.reset()
        assert idx.dirty == 0

    def test_dirty_counts_upsert_transitively(self):
        """Upsert delegates to add/remove, so dirty counts transitively."""
        idx = NphdIndex(max_dim=64)
        vec1 = np.random.randint(0, 256, size=8, dtype=np.uint8)
        vec2 = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec1)
        assert idx.dirty == 1

        # upsert existing key: remove(1) + add(1) = +2
        idx.upsert(1, vec2)
        assert idx.dirty >= 2  # at least the original add + the upsert's writes

    def test_dirty_truthy_check(self):
        """Dirty counter supports truthy checks like `if idx.dirty:`."""
        idx = NphdIndex(max_dim=64)
        assert not idx.dirty  # 0 is falsy

        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec)
        assert idx.dirty  # non-zero is truthy

    def test_copy_starts_clean(self):
        """A copy of a dirty index starts with dirty=0."""
        idx = NphdIndex(max_dim=64)
        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec)
        assert idx.dirty == 1

        copy = idx.copy()
        assert copy.dirty == 0


# === ShardedIndex dirty counter ===


class TestShardedIndexDirty:
    """Tests for ShardedIndex.dirty property."""

    def test_dirty_starts_at_zero(self, tmp_path):
        idx = ShardedIndex(ndim=32, path=tmp_path)
        assert idx.dirty == 0

    def test_dirty_increments_on_single_add(self, tmp_path):
        idx = ShardedIndex(ndim=32, path=tmp_path)
        vec = np.random.rand(32).astype(np.float32)
        idx.add(1, vec)
        assert idx.dirty == 1

    def test_dirty_increments_on_batch_add(self, tmp_path):
        idx = ShardedIndex(ndim=32, path=tmp_path)
        vecs = np.random.rand(10, 32).astype(np.float32)
        idx.add(list(range(10)), vecs)
        assert idx.dirty == 10

    def test_dirty_increments_on_single_remove(self, tmp_path):
        idx = ShardedIndex(ndim=32, path=tmp_path)
        vec = np.random.rand(32).astype(np.float32)
        idx.add(1, vec)
        idx.remove(1)
        assert idx.dirty == 2  # 1 add + 1 remove

    def test_dirty_increments_on_batch_remove(self, tmp_path):
        idx = ShardedIndex(ndim=32, path=tmp_path)
        vecs = np.random.rand(5, 32).astype(np.float32)
        idx.add(list(range(5)), vecs)
        idx.remove([0, 1, 2])
        assert idx.dirty == 8  # 5 adds + 3 removes

    def test_dirty_resets_on_save(self, tmp_path):
        idx = ShardedIndex(ndim=32, path=tmp_path)
        vecs = np.random.rand(10, 32).astype(np.float32)
        idx.add(list(range(10)), vecs)
        assert idx.dirty == 10

        idx.save()
        assert idx.dirty == 0

    def test_dirty_resets_on_save_empty(self, tmp_path):
        """Save of empty index still resets dirty (e.g. after removes)."""
        idx = ShardedIndex(ndim=32, path=tmp_path)
        idx.save()
        assert idx.dirty == 0

    def test_dirty_resets_on_reset(self, tmp_path):
        idx = ShardedIndex(ndim=32, path=tmp_path)
        vecs = np.random.rand(10, 32).astype(np.float32)
        idx.add(list(range(10)), vecs)
        assert idx.dirty == 10

        idx.reset()
        assert idx.dirty == 0

    def test_dirty_zero_after_construction_with_existing_shards(self, tmp_path):
        """Loading existing shards starts with dirty=0."""
        idx = ShardedIndex(ndim=32, path=tmp_path)
        vecs = np.random.rand(10, 32).astype(np.float32)
        idx.add(list(range(10)), vecs)
        idx.save()

        # Reopen — constructor loads existing shards
        idx2 = ShardedIndex(ndim=32, path=tmp_path)
        assert idx2.dirty == 0

    def test_dirty_survives_shard_rotation(self, tmp_path):
        """Shard rotation does NOT reset the dirty counter."""
        idx = ShardedIndex(ndim=32, path=tmp_path, shard_size=500)

        for i in range(100):
            vec = np.random.rand(32).astype(np.float32)
            idx.add(i, vec)

        assert idx.shard_count >= 2, "Expected shard rotation"
        assert idx.dirty == 100  # all 100 adds still counted

    def test_dirty_counts_upsert_transitively(self, tmp_path):
        """Upsert delegates to remove+add, dirty counts both."""
        idx = ShardedIndex(ndim=32, path=tmp_path)
        vec1 = np.random.rand(32).astype(np.float32)
        vec2 = np.random.rand(32).astype(np.float32)
        idx.add(1, vec1)
        assert idx.dirty == 1

        idx.upsert(1, vec2)
        # upsert(1) = remove(1) + add(1) = +2
        assert idx.dirty == 3

    def test_dirty_counts_add_once(self, tmp_path):
        """add_once delegates to add for new keys."""
        idx = ShardedIndex(ndim=32, path=tmp_path)
        vec = np.random.rand(32).astype(np.float32)
        idx.add_once(1, vec)
        assert idx.dirty == 1

        # Adding same key again — skipped, no mutation
        idx.add_once(1, vec)
        assert idx.dirty == 1  # unchanged

    def test_dirty_flush_every_n_pattern(self, tmp_path):
        """Demonstrates the 'flush every N writes' pattern."""
        idx = ShardedIndex(ndim=32, path=tmp_path)
        flush_threshold = 5

        for i in range(12):
            vec = np.random.rand(32).astype(np.float32)
            idx.add(i, vec)
            if idx.dirty >= flush_threshold:
                idx.save()
                assert idx.dirty == 0

    def test_dirty_read_only_always_zero(self, tmp_path):
        """Read-only indexes always return dirty=0."""
        # Create and save an index
        idx = ShardedIndex(ndim=32, path=tmp_path)
        vecs = np.random.rand(10, 32).astype(np.float32)
        idx.add(list(range(10)), vecs)
        idx.save()

        # Open read-only
        ro_idx = ShardedIndex(ndim=32, path=tmp_path, read_only=True)
        assert ro_idx.dirty == 0


# === ShardedIndex128 dirty counter ===


class TestShardedIndex128Dirty:
    """Tests for ShardedIndex128.dirty property."""

    def test_dirty_increments_and_resets(self, tmp_path):
        idx = ShardedIndex128(ndim=32, path=tmp_path)
        key = b"\x01" * 16
        vec = np.random.rand(32).astype(np.float32)

        idx.add(key, vec)
        assert idx.dirty == 1

        idx.remove(key)
        assert idx.dirty == 2

        idx.save()
        assert idx.dirty == 0

    def test_dirty_batch_operations(self, tmp_path):
        idx = ShardedIndex128(ndim=32, path=tmp_path)
        keys = [i.to_bytes(16, "big") for i in range(5)]
        keys_arr = np.array(keys, dtype=np.dtype("V16"))
        vecs = np.random.rand(5, 32).astype(np.float32)

        idx.add(keys_arr, vecs)
        assert idx.dirty == 5

    def test_dirty_read_only(self, tmp_path):
        idx = ShardedIndex128(ndim=32, path=tmp_path)
        key = b"\x01" * 16
        vec = np.random.rand(32).astype(np.float32)
        idx.add(key, vec)
        idx.save()

        ro_idx = ShardedIndex128(ndim=32, path=tmp_path, read_only=True)
        assert ro_idx.dirty == 0


# === ShardedNphdIndex dirty counter ===


class TestShardedNphdIndexDirty:
    """Tests for ShardedNphdIndex.dirty property."""

    def test_dirty_starts_at_zero(self, tmp_path):
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        assert idx.dirty == 0

    def test_dirty_increments_on_add(self, tmp_path):
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec)
        assert idx.dirty == 1

    def test_dirty_increments_on_batch_add(self, tmp_path):
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        vecs = np.random.randint(0, 256, size=(5, 8), dtype=np.uint8)
        idx.add(list(range(5)), vecs)
        assert idx.dirty == 5

    def test_dirty_increments_on_remove(self, tmp_path):
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec)
        idx.remove(1)
        assert idx.dirty == 2

    def test_dirty_resets_on_save(self, tmp_path):
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec)
        idx.save()
        assert idx.dirty == 0

    def test_dirty_resets_on_reset(self, tmp_path):
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec)
        idx.reset()
        assert idx.dirty == 0

    def test_dirty_survives_shard_rotation(self, tmp_path):
        """Shard rotation does NOT reset dirty counter."""
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path, shard_size=500)

        for i in range(100):
            vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
            idx.add(i, vec)

        assert idx.shard_count >= 2, "Expected shard rotation"
        assert idx.dirty == 100

    def test_dirty_upsert(self, tmp_path):
        """Upsert delegates to remove+add, dirty counts transitively."""
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        vec1 = np.random.randint(0, 256, size=8, dtype=np.uint8)
        vec2 = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec1)
        idx.upsert(1, vec2)
        # upsert(1) = remove(1) + add(1) = +2
        assert idx.dirty == 3

    def test_dirty_read_only(self, tmp_path):
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(1, vec)
        idx.save()

        ro_idx = ShardedNphdIndex(max_dim=64, path=tmp_path, read_only=True)
        assert ro_idx.dirty == 0


# === ShardedNphdIndex128 dirty counter ===


class TestShardedNphdIndex128Dirty:
    """Tests for ShardedNphdIndex128.dirty property."""

    def test_dirty_increments_and_resets(self, tmp_path):
        idx = ShardedNphdIndex128(max_dim=64, path=tmp_path)
        key = b"\x01" * 16
        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)

        idx.add(key, vec)
        assert idx.dirty == 1

        idx.remove(key)
        assert idx.dirty == 2

        idx.save()
        assert idx.dirty == 0

    def test_dirty_read_only(self, tmp_path):
        idx = ShardedNphdIndex128(max_dim=64, path=tmp_path)
        key = b"\x01" * 16
        vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
        idx.add(key, vec)
        idx.save()

        ro_idx = ShardedNphdIndex128(max_dim=64, path=tmp_path, read_only=True)
        assert ro_idx.dirty == 0
