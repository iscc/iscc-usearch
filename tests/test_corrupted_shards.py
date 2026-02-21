"""
Test graceful error handling for corrupted shard files.

Verifies that corrupted or truncated .usearch shard files are handled gracefully
instead of crashing the process. Corrupted shards should be skipped with warnings,
and the index should remain operational with whatever valid shards remain.
"""

from unittest.mock import patch

import numpy as np
import pytest
from usearch.index import Index

from iscc_usearch import CorruptedShardError, ShardedIndex, ShardedNphdIndex


class TestCorruptedShardError:
    """Test the CorruptedShardError exception class."""

    def test_basic_creation(self):
        err = CorruptedShardError("/path/to/shard_000.usearch")
        assert "shard_000.usearch" in str(err)
        assert err.path.name == "shard_000.usearch"

    def test_with_reason(self):
        err = CorruptedShardError("/path/to/shard_000.usearch", "metadata unreadable")
        assert "metadata unreadable" in str(err)
        assert "shard_000.usearch" in str(err)

    def test_is_exception(self):
        assert issubclass(CorruptedShardError, Exception)

    def test_catchable(self):
        with pytest.raises(CorruptedShardError):
            raise CorruptedShardError("/path/to/shard.usearch", "test")


class TestShardedIndexCorruptedShards:
    """Test ShardedIndex behavior with corrupted shard files."""

    def test_corrupted_single_shard_creates_empty_index(self, tmp_path):
        """When the only shard is corrupted, constructor succeeds with size=0."""
        # Create a valid index and save it
        idx = ShardedIndex(ndim=64, path=tmp_path)
        vectors = np.random.rand(5, 64).astype(np.float32)
        idx.add(list(range(5)), vectors)
        idx.save()

        # Corrupt the shard file
        shard_files = list(tmp_path.glob("shard_*.usearch"))
        assert len(shard_files) == 1
        shard_files[0].write_bytes(b"CORRUPTED DATA")

        # Reopen — should succeed with empty index
        idx2 = ShardedIndex(ndim=64, path=tmp_path)
        assert len(idx2) == 0

    def _create_multi_shard_index(self, tmp_path, ndim=64):
        """Helper to create an index with multiple shards via individual adds."""
        idx = ShardedIndex(ndim=ndim, path=tmp_path, shard_size=100)
        for i in range(100):
            idx.add(i, np.random.rand(ndim).astype(np.float32))
        idx.save()
        shard_files = sorted(tmp_path.glob("shard_*.usearch"))
        assert len(shard_files) >= 2, f"Expected multiple shards, got {len(shard_files)}"
        return idx, shard_files

    def test_corrupted_view_shard_skipped(self, tmp_path):
        """When a view shard is corrupted, it's skipped and valid shards remain."""
        idx, shard_files = self._create_multi_shard_index(tmp_path)
        total_before = len(idx)

        # Corrupt the first shard (will be a view shard on reopen)
        shard_files[0].write_bytes(b"CORRUPTED DATA")

        # Reopen — should skip the corrupted shard
        idx2 = ShardedIndex(ndim=64, path=tmp_path)
        assert len(idx2) > 0
        assert len(idx2) < total_before

    def test_corrupted_active_shard_creates_fresh(self, tmp_path):
        """When the active (last) shard is corrupted, a fresh shard is created."""
        idx, shard_files = self._create_multi_shard_index(tmp_path)

        # Corrupt the last shard (active shard on reopen)
        shard_files[-1].write_bytes(b"CORRUPTED DATA")

        # Reopen — should create a fresh active shard, keep view shards
        idx2 = ShardedIndex(ndim=64, path=tmp_path)
        assert len(idx2) > 0  # View shards still have data
        assert idx2._active_shard is not None
        assert idx2._active_shard_path is None  # Fresh shard, no tracked path

    def test_all_shards_corrupted_creates_empty(self, tmp_path):
        """When all shards are corrupted, constructor succeeds with size=0."""
        idx, _ = self._create_multi_shard_index(tmp_path)

        # Corrupt all shards
        for shard_file in tmp_path.glob("shard_*.usearch"):
            shard_file.write_bytes(b"CORRUPTED DATA")

        # Reopen — should succeed with empty index
        idx2 = ShardedIndex(ndim=64, path=tmp_path)
        assert len(idx2) == 0

    def test_corrupted_shard_read_only_skipped(self, tmp_path):
        """In read-only mode, corrupted shards are skipped."""
        idx, shard_files = self._create_multi_shard_index(tmp_path)

        # Corrupt the first shard
        shard_files[0].write_bytes(b"CORRUPTED DATA")

        # Reopen read-only — should skip the corrupted shard
        idx2 = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
        assert len(idx2) > 0

    def test_truncated_shard_handled(self, tmp_path):
        """A truncated shard file (partial write) is handled gracefully."""
        # Create a valid index and save it
        idx = ShardedIndex(ndim=64, path=tmp_path)
        vectors = np.random.rand(10, 64).astype(np.float32)
        idx.add(list(range(10)), vectors)
        idx.save()

        shard_files = list(tmp_path.glob("shard_*.usearch"))
        assert len(shard_files) == 1

        # Truncate the shard file (keep first 10 bytes)
        original = shard_files[0].read_bytes()
        shard_files[0].write_bytes(original[:10])

        # Reopen — should handle gracefully
        idx2 = ShardedIndex(ndim=64, path=tmp_path)
        assert idx2 is not None

    def test_empty_shard_file_handled(self, tmp_path):
        """An empty shard file (0 bytes) is handled gracefully."""
        # Create a valid index and save it
        idx = ShardedIndex(ndim=64, path=tmp_path)
        vectors = np.random.rand(5, 64).astype(np.float32)
        idx.add(list(range(5)), vectors)
        idx.save()

        shard_files = list(tmp_path.glob("shard_*.usearch"))
        assert len(shard_files) == 1

        # Make the shard file empty
        shard_files[0].write_bytes(b"")

        # Reopen — should handle gracefully
        idx2 = ShardedIndex(ndim=64, path=tmp_path)
        assert len(idx2) == 0

    def test_resolve_config_skips_corrupted_first_shard(self, tmp_path):
        """Config auto-detection falls back to valid shards if first is corrupted."""
        idx, shard_files = self._create_multi_shard_index(tmp_path)

        # Corrupt the first shard (used for config auto-detection)
        shard_files[0].write_bytes(b"CORRUPTED DATA")

        # Reopen without specifying ndim — should auto-detect from second shard
        idx2 = ShardedIndex(path=tmp_path)
        assert idx2.ndim == 64

    def test_can_add_after_corrupted_shard_recovery(self, tmp_path):
        """After recovering from corrupted shards, the index is fully usable."""
        # Create a valid index and save it
        idx = ShardedIndex(ndim=64, path=tmp_path)
        vectors = np.random.rand(5, 64).astype(np.float32)
        idx.add(list(range(5)), vectors)
        idx.save()

        # Corrupt the shard
        for shard_file in tmp_path.glob("shard_*.usearch"):
            shard_file.write_bytes(b"CORRUPTED DATA")

        # Reopen — should succeed with empty index
        idx2 = ShardedIndex(ndim=64, path=tmp_path)
        assert len(idx2) == 0

        # Should be fully usable for new additions
        new_vectors = np.random.rand(3, 64).astype(np.float32)
        idx2.add(list(range(3)), new_vectors)
        assert len(idx2) == 3

        # Verify search works
        results = idx2.search(new_vectors[0:1], 1)
        assert results is not None


class TestShardedNphdIndexCorruptedShards:
    """Test ShardedNphdIndex behavior with corrupted shard files."""

    def test_corrupted_shard_creates_empty_nphd_index(self, tmp_path):
        """ShardedNphdIndex handles corrupted shards and succeeds with size=0."""
        # Create a valid NPHD index
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        vectors = [np.random.bytes(8) for _ in range(5)]
        idx.add(list(range(5)), vectors)
        idx.save()

        # Corrupt the shard
        for shard_file in tmp_path.glob("shard_*.usearch"):
            shard_file.write_bytes(b"CORRUPTED DATA")

        # Reopen — should succeed with empty index
        idx2 = ShardedNphdIndex(max_dim=64, path=tmp_path)
        assert len(idx2) == 0

    def test_resolve_max_dim_skips_corrupted(self, tmp_path):
        """max_dim auto-detection falls back to valid shards."""
        # Create an NPHD index with multiple shards via individual adds
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path, shard_size=100)
        for i in range(100):
            idx.add(i, [np.random.bytes(8)])
        idx.save()

        shard_files = sorted(tmp_path.glob("shard_*.usearch"))
        assert len(shard_files) >= 2, f"Expected multiple shards, got {len(shard_files)}"

        # Corrupt the first shard
        shard_files[0].write_bytes(b"CORRUPTED DATA")

        # Reopen without specifying max_dim — should auto-detect from second shard
        idx2 = ShardedNphdIndex(path=tmp_path)
        assert idx2.max_dim == 64

    def test_all_shards_corrupted_with_max_dim_succeeds(self, tmp_path):
        """When max_dim is explicitly provided, all-corrupted shards don't prevent creation."""
        # Create a valid index
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        vectors = [np.random.bytes(8) for _ in range(5)]
        idx.add(list(range(5)), vectors)
        idx.save()

        # Corrupt all shards
        for shard_file in tmp_path.glob("shard_*.usearch"):
            shard_file.write_bytes(b"CORRUPTED DATA")

        # Reopen with explicit max_dim — should succeed
        idx2 = ShardedNphdIndex(max_dim=64, path=tmp_path)
        assert len(idx2) == 0

    def test_all_shards_corrupted_without_max_dim_raises(self, tmp_path):
        """When all shards corrupted and no max_dim, raise ValueError."""
        # Create a valid index
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        vectors = [np.random.bytes(8) for _ in range(5)]
        idx.add(list(range(5)), vectors)
        idx.save()

        # Corrupt all shards
        for shard_file in tmp_path.glob("shard_*.usearch"):
            shard_file.write_bytes(b"CORRUPTED DATA")

        # Reopen without max_dim — should raise
        with pytest.raises(ValueError, match="all shard metadata unreadable"):
            ShardedNphdIndex(path=tmp_path)

    def test_can_add_after_nphd_corrupted_recovery(self, tmp_path):
        """After recovering from corrupted shards, NPHD index is fully usable."""
        # Create and corrupt
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        vectors = [np.random.bytes(8) for _ in range(5)]
        idx.add(list(range(5)), vectors)
        idx.save()
        for shard_file in tmp_path.glob("shard_*.usearch"):
            shard_file.write_bytes(b"CORRUPTED DATA")

        # Recover
        idx2 = ShardedNphdIndex(max_dim=64, path=tmp_path)
        assert len(idx2) == 0

        # Should be fully usable
        new_vectors = [np.random.bytes(8) for _ in range(3)]
        idx2.add(list(range(3)), new_vectors)
        assert len(idx2) == 3


class TestMetadataReturnsNone:
    """Test paths where Index.metadata() returns None instead of raising."""

    def test_restore_shard_metadata_none_base(self, tmp_path):
        """ShardedIndex._restore_shard returns None when metadata() returns None."""
        idx = ShardedIndex(ndim=64, path=tmp_path)
        vectors = np.random.rand(5, 64).astype(np.float32)
        idx.add(list(range(5)), vectors)
        idx.save()

        with patch.object(Index, "metadata", return_value=None):
            idx2 = ShardedIndex(ndim=64, path=tmp_path)
        assert len(idx2) == 0

    def test_restore_shard_metadata_none_nphd(self, tmp_path):
        """ShardedNphdIndex._restore_shard returns None when metadata() returns None."""
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        vectors = [np.random.bytes(8) for _ in range(5)]
        idx.add(list(range(5)), vectors)
        idx.save()

        with patch.object(Index, "metadata", return_value=None):
            idx2 = ShardedNphdIndex(max_dim=64, path=tmp_path)
        assert len(idx2) == 0

    def test_restore_shard_metadata_none_uuid(self, tmp_path):
        """ShardedIndex128._restore_shard returns None when metadata() returns None."""
        from iscc_usearch.sharded import ShardedIndex128

        idx = ShardedIndex128(ndim=64, path=tmp_path)
        vectors = np.random.rand(5, 64).astype(np.float32)
        keys = [i.to_bytes(16, "big") for i in range(5)]
        idx.add(keys, vectors)
        idx.save()

        with patch.object(Index, "metadata", return_value=None):
            idx2 = ShardedIndex128(ndim=64, path=tmp_path)
        assert len(idx2) == 0

    def test_restore_shard_metadata_none_nphd_uuid(self, tmp_path):
        """ShardedNphdIndex128._restore_shard returns None when metadata() returns None."""
        from iscc_usearch.sharded_nphd import ShardedNphdIndex128

        idx = ShardedNphdIndex128(max_dim=64, path=tmp_path)
        keys = [i.to_bytes(16, "big") for i in range(5)]
        vectors = [np.random.bytes(8) for _ in range(5)]
        idx.add(keys, vectors)
        idx.save()

        with patch.object(Index, "metadata", return_value=None):
            idx2 = ShardedNphdIndex128(max_dim=64, path=tmp_path)
        assert len(idx2) == 0

    def test_resolve_config_metadata_none_all_shards(self, tmp_path):
        """_resolve_config raises when all shard metadata returns None and ndim not given."""
        idx = ShardedIndex(ndim=64, path=tmp_path)
        vectors = np.random.rand(5, 64).astype(np.float32)
        idx.add(list(range(5)), vectors)
        idx.save()

        with patch.object(Index, "metadata", return_value=None):
            with pytest.raises(ValueError, match="all shard metadata unreadable"):
                ShardedIndex(path=tmp_path)

    def test_resolve_config_metadata_none_with_ndim(self, tmp_path):
        """_resolve_config uses provided ndim when all shard metadata returns None."""
        idx = ShardedIndex(ndim=64, path=tmp_path)
        vectors = np.random.rand(5, 64).astype(np.float32)
        idx.add(list(range(5)), vectors)
        idx.save()

        with patch.object(Index, "metadata", return_value=None):
            idx2 = ShardedIndex(ndim=64, path=tmp_path)
        assert len(idx2) == 0

    def test_resolve_max_dim_metadata_none_all_shards(self, tmp_path):
        """_resolve_max_dim raises when all shard metadata returns None and max_dim not given."""
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        vectors = [np.random.bytes(8) for _ in range(5)]
        idx.add(list(range(5)), vectors)
        idx.save()

        with patch.object(Index, "metadata", return_value=None):
            with pytest.raises(ValueError, match="all shard metadata unreadable"):
                ShardedNphdIndex(path=tmp_path)


class TestAllCorruptedBloomCreation:
    """Test bloom filter creation when all shards are corrupted."""

    def test_all_corrupted_read_only_no_bloom_creates_bloom(self, tmp_path):
        """When all shards corrupted (read-only) and no bloom file, new bloom is created."""
        idx = ShardedIndex(ndim=64, path=tmp_path, bloom_filter=True)
        vectors = np.random.rand(5, 64).astype(np.float32)
        idx.add(list(range(5)), vectors)
        idx.save()

        # Corrupt shard AND remove bloom file
        for shard_file in tmp_path.glob("shard_*.usearch"):
            shard_file.write_bytes(b"CORRUPTED DATA")
        bloom_path = tmp_path / "bloom.isbf"
        if bloom_path.exists():
            bloom_path.unlink()

        # Read-only with all corrupted + no bloom → falls to all-corrupted path with bloom creation
        idx2 = ShardedIndex(ndim=64, path=tmp_path, bloom_filter=True, read_only=True)
        assert len(idx2) == 0
        assert idx2._bloom is not None


class TestRestoreShardExceptionPaths128:
    """Test exception paths in _restore_shard for UUID variants."""

    def test_restore_shard_metadata_exception_uuid(self, tmp_path):
        """ShardedIndex128._restore_shard handles metadata exception."""
        from iscc_usearch.sharded import ShardedIndex128

        idx = ShardedIndex128(ndim=64, path=tmp_path)
        keys = [i.to_bytes(16, "big") for i in range(5)]
        vectors = np.random.rand(5, 64).astype(np.float32)
        idx.add(keys, vectors)
        idx.save()

        # Corrupt shard so metadata() raises
        for shard_file in tmp_path.glob("shard_*.usearch"):
            shard_file.write_bytes(b"CORRUPTED")

        idx2 = ShardedIndex128(ndim=64, path=tmp_path)
        assert len(idx2) == 0

    def test_restore_shard_metadata_exception_nphd_uuid(self, tmp_path):
        """ShardedNphdIndex128._restore_shard handles metadata exception."""
        from iscc_usearch.sharded_nphd import ShardedNphdIndex128

        idx = ShardedNphdIndex128(max_dim=64, path=tmp_path)
        keys = [i.to_bytes(16, "big") for i in range(5)]
        vectors = [np.random.bytes(8) for _ in range(5)]
        idx.add(keys, vectors)
        idx.save()

        # Corrupt shard so metadata() raises
        for shard_file in tmp_path.glob("shard_*.usearch"):
            shard_file.write_bytes(b"CORRUPTED")

        idx2 = ShardedNphdIndex128(max_dim=64, path=tmp_path)
        assert len(idx2) == 0

    def test_restore_shard_load_fails_nphd(self, tmp_path):
        """ShardedNphdIndex._restore_shard returns None when load() raises."""
        idx = ShardedNphdIndex(max_dim=64, path=tmp_path)
        vectors = [np.random.bytes(8) for _ in range(5)]
        idx.add(list(range(5)), vectors)
        idx.save()

        def failing_load(self, *args, **kwargs):
            raise RuntimeError("simulated HNSW graph corruption")

        with patch.object(Index, "load", failing_load):
            idx2 = ShardedNphdIndex(max_dim=64, path=tmp_path)
        assert len(idx2) == 0


class TestLoadViewExceptionPaths:
    """Test paths where Index.load() or Index.view() raises after metadata succeeds."""

    def _make_index_with_shard(self, tmp_path, cls=ShardedIndex, **kwargs):
        """Create an index with one saved shard."""
        idx = cls(path=tmp_path, **kwargs)
        return idx

    def test_restore_shard_load_fails_base(self, tmp_path):
        """ShardedIndex._restore_shard returns None when load() raises."""
        idx = ShardedIndex(ndim=64, path=tmp_path)
        vectors = np.random.rand(5, 64).astype(np.float32)
        idx.add(list(range(5)), vectors)
        idx.save()

        # Patch Index.load to raise after metadata succeeds
        def failing_load(self, *args, **kwargs):
            raise RuntimeError("simulated HNSW graph corruption")

        with patch.object(Index, "load", failing_load):
            idx2 = ShardedIndex(ndim=64, path=tmp_path)
        assert len(idx2) == 0

    def test_restore_shard_view_fails_base(self, tmp_path):
        """ShardedIndex._restore_shard returns None when view() raises."""
        idx = ShardedIndex(ndim=64, path=tmp_path)
        vectors = np.random.rand(5, 64).astype(np.float32)
        idx.add(list(range(5)), vectors)
        idx.save()

        def failing_view(self, *args, **kwargs):
            raise RuntimeError("simulated mmap corruption")

        with patch.object(Index, "view", failing_view):
            # In read-only mode, all shards use view
            idx2 = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
        assert len(idx2) == 0

    def test_all_shards_corrupted_read_only_empty(self, tmp_path):
        """Read-only mode with all shards corrupted creates empty index."""
        idx = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)
        for i in range(100):
            idx.add(i, np.random.rand(64).astype(np.float32))
        idx.save()

        # Corrupt all shards
        for shard_file in tmp_path.glob("shard_*.usearch"):
            shard_file.write_bytes(b"CORRUPTED DATA")

        # Read-only with all corrupted — should fall back to empty (viewed_indexes empty)
        idx2 = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
        assert len(idx2) == 0

    def test_read_only_all_corrupted_last_shard_from_views(self, tmp_path):
        """Read-only mode uses last viewed shard for config when some are valid."""
        idx = ShardedIndex(ndim=64, path=tmp_path, shard_size=100)
        for i in range(100):
            idx.add(i, np.random.rand(64).astype(np.float32))
        idx.save()

        shard_files = sorted(tmp_path.glob("shard_*.usearch"))
        assert len(shard_files) >= 2

        # Corrupt only the first shard
        shard_files[0].write_bytes(b"CORRUPTED DATA")

        # Read-only — should pick last_shard from viewed_indexes
        idx2 = ShardedIndex(ndim=64, path=tmp_path, read_only=True)
        assert len(idx2) > 0
        assert idx2._active_shard is None  # read-only, no active shard
