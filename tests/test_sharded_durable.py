"""
Test durable write path for ShardedIndex shard persistence.

Verifies that all writable shard saves go through buffer + durable_write,
bloom/tombstone ordering is correct, and crash recovery is resilient.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from iscc_usearch.sharded import ShardedIndex


def test_save_uses_durable_write(tmp_path):
    """save() persists the active shard via durable_write, not direct .save(path)."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    idx.add(list(range(5)), np.random.rand(5, 32).astype(np.float32))

    calls = []
    original_durable_write = __import__("iscc_usearch.utils", fromlist=["durable_write"]).durable_write

    def tracking_durable_write(data, target):
        calls.append(Path(target).name)
        return original_durable_write(data, target)

    with patch("iscc_usearch.sharded.durable_write", side_effect=tracking_durable_write):
        idx.save()

    shard_calls = [c for c in calls if c.endswith(".usearch")]
    assert len(shard_calls) == 1
    assert shard_calls[0] == "shard_000.usearch"


def test_rotation_uses_durable_write(tmp_path):
    """_rotate_shard() persists the shard via durable_write."""
    idx = ShardedIndex(ndim=32, path=tmp_path, shard_size=500)

    calls = []
    original_durable_write = __import__("iscc_usearch.utils", fromlist=["durable_write"]).durable_write

    def tracking_durable_write(data, target):
        calls.append(Path(target).name)
        return original_durable_write(data, target)

    with patch("iscc_usearch.sharded.durable_write", side_effect=tracking_durable_write):
        for i in range(100):
            idx.add(i, np.random.rand(32).astype(np.float32))

    shard_calls = [c for c in calls if c.endswith(".usearch")]
    assert len(shard_calls) >= 1, "At least one shard rotation should use durable_write"


def test_compact_uses_durable_write(tmp_path):
    """compact() persists rebuilt shards via durable_write."""
    idx = ShardedIndex(ndim=32, path=tmp_path, shard_size=500)
    for i in range(100):
        idx.add(i, np.random.rand(32).astype(np.float32))
    idx.save()

    # Remove some keys to create tombstones
    idx.remove([0, 1, 2])

    calls = []
    original_durable_write = __import__("iscc_usearch.utils", fromlist=["durable_write"]).durable_write

    def tracking_durable_write(data, target):
        calls.append(Path(target).name)
        return original_durable_write(data, target)

    with patch("iscc_usearch.sharded.durable_write", side_effect=tracking_durable_write):
        idx.compact()

    shard_calls = [c for c in calls if c.endswith(".usearch")]
    assert len(shard_calls) >= 1, "compact() should use durable_write for rebuilt shards"


def test_original_shard_intact_on_durable_write_failure(tmp_path):
    """If durable_write fails, the original shard file is not corrupted."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    idx.add(list(range(5)), np.random.rand(5, 32).astype(np.float32))
    idx.save()

    shard_path = tmp_path / "shard_000.usearch"
    original_size = shard_path.stat().st_size
    original_data = shard_path.read_bytes()

    # Add more data, then make save fail
    idx.add(list(range(10, 15)), np.random.rand(5, 32).astype(np.float32))

    def failing_durable_write(data, target):
        raise OSError("Simulated write failure")

    with patch("iscc_usearch.sharded.durable_write", side_effect=failing_durable_write):
        with pytest.raises(OSError, match="Simulated write failure"):
            idx.save()

    # Original file should be unchanged
    assert shard_path.stat().st_size == original_size
    assert shard_path.read_bytes() == original_data


def test_rotation_persistence_order(tmp_path):
    """_rotate_shard() persists bloom → shard → tombstones (shard before tombstones)."""
    idx = ShardedIndex(ndim=32, path=tmp_path, shard_size=500)

    order = []
    original_persist_bloom = ShardedIndex._persist_bloom
    original_persist_tombstones = ShardedIndex._persist_tombstones
    original_durable_write = __import__("iscc_usearch.utils", fromlist=["durable_write"]).durable_write

    def tracking_bloom(self):
        order.append("bloom")
        return original_persist_bloom(self)

    def tracking_tombstones(self):
        order.append("tombstones")
        return original_persist_tombstones(self)

    def tracking_durable_write(data, target):
        if str(target).endswith(".usearch"):
            order.append("shard")
        return original_durable_write(data, target)

    with (
        patch.object(ShardedIndex, "_persist_bloom", tracking_bloom),
        patch.object(ShardedIndex, "_persist_tombstones", tracking_tombstones),
        patch("iscc_usearch.sharded.durable_write", side_effect=tracking_durable_write),
    ):
        for i in range(100):
            idx.add(i, np.random.rand(32).astype(np.float32))

    bloom_indices = [i for i, x in enumerate(order) if x == "bloom"]
    shard_indices = [i for i, x in enumerate(order) if x == "shard"]
    tombstone_indices = [i for i, x in enumerate(order) if x == "tombstones"]
    assert bloom_indices, "Bloom should be persisted during rotation"
    assert shard_indices, "Shard should be persisted during rotation"
    assert tombstone_indices, "Tombstones should be persisted during rotation"
    for b, s in zip(bloom_indices, shard_indices):
        assert b < s, f"Bloom (index {b}) must precede shard (index {s})"
    for s, t in zip(shard_indices, tombstone_indices):
        assert s < t, f"Shard (index {s}) must precede tombstones (index {t})"


def test_bloom_rebuild_on_corrupt_file(tmp_path):
    """Corrupt bloom file triggers rebuild from shard keys."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    idx.add(list(range(10)), np.random.rand(10, 32).astype(np.float32))
    idx.save()

    # Corrupt the bloom file
    bloom_path = tmp_path / "bloom.isbf"
    bloom_path.write_bytes(b"corrupt data")

    # Reload — should rebuild bloom without error
    idx2 = ShardedIndex(ndim=32, path=tmp_path)
    assert idx2._bloom is not None
    assert idx2._bloom.count == 10
    assert idx2.contains(0)
    assert idx2.contains(9)
    assert not idx2.contains(999)


def test_tombstone_file_missing_on_reload(tmp_path):
    """Missing tombstone file on reload doesn't crash — tombstoned keys reappear."""
    idx = ShardedIndex(ndim=32, path=tmp_path, shard_size=500)
    for i in range(100):
        idx.add(i, np.random.rand(32).astype(np.float32))
    idx.save()

    # Remove some keys and save
    idx.remove([0, 1, 2])
    idx.save()

    # Verify tombstone file exists
    tombstone_path = tmp_path / "tombstones.npy"
    assert tombstone_path.exists()

    # Delete tombstone file
    tombstone_path.unlink()

    # Reload — should not crash, tombstoned keys reappear
    idx2 = ShardedIndex(ndim=32, path=tmp_path)
    assert len(idx2) >= 100  # All keys including formerly tombstoned ones
    assert idx2.contains(0)
    assert idx2.contains(1)
    assert idx2.contains(2)


def test_stale_tmp_shard_cleaned_up_on_load(tmp_path):
    """Stale .tmp files from interrupted durable_write are cleaned on load."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    idx.add(list(range(5)), np.random.rand(5, 32).astype(np.float32))
    idx.save()

    # Simulate interrupted durable_write leaving a .tmp file
    stale_tmp = tmp_path / "shard_000.usearch.tmp"
    stale_tmp.write_bytes(b"incomplete write")

    # Reload — .tmp should be cleaned up
    idx2 = ShardedIndex(ndim=32, path=tmp_path)
    assert not stale_tmp.exists()
    assert len(idx2) == 5


def test_save_progress_callback_preserved(tmp_path):
    """save(progress=callback) passes the progress callback through to buffer serialization."""
    idx = ShardedIndex(ndim=32, path=tmp_path)
    idx.add(list(range(5)), np.random.rand(5, 32).astype(np.float32))

    progress_calls = []

    def progress_cb(done: int, total: int) -> bool:
        progress_calls.append((done, total))
        return True

    idx.save(progress=progress_cb)
    assert len(progress_calls) > 0


def test_no_direct_save_path_in_sharded(tmp_path):
    """Audit: no writable shard persistence bypasses durable_write."""
    import ast

    sharded_path = Path(__file__).parent.parent / "src" / "iscc_usearch" / "sharded.py"
    source = sharded_path.read_text()
    tree = ast.parse(source)

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match pattern: <something>.save(<string_arg>) where string_arg is a str conversion
        if not isinstance(func, ast.Attribute) or func.attr != "save":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        # Check if the first arg is str(...) or a string literal — a path argument
        if isinstance(first_arg, ast.Call) and isinstance(first_arg.func, ast.Name) and first_arg.func.id == "str":
            violations.append(node.lineno)
        elif isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            violations.append(node.lineno)

    assert violations == [], f"Direct .save(path) calls found at lines: {violations}"


def test_bloom_not_rebuilt_when_disabled(tmp_path):
    """When bloom_filter=False, missing bloom file doesn't trigger rebuild."""
    idx = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=False)
    idx.add(list(range(5)), np.random.rand(5, 32).astype(np.float32))
    idx.save()

    idx2 = ShardedIndex(ndim=32, path=tmp_path, bloom_filter=False)
    assert idx2._bloom is None
    assert idx2.contains(0)


def test_tombstone_removal_safe_across_crash(tmp_path):
    """Tombstones are persisted after shard, so removed tombstones can't expose stale keys."""
    idx = ShardedIndex(ndim=32, path=tmp_path, shard_size=500)

    # Build enough data to have view shards
    for i in range(100):
        idx.add(i, np.random.rand(32).astype(np.float32))
    idx.save()

    # Remove key 0 (tombstone it) and persist the tombstone to disk
    idx.remove(0)
    idx.save()
    assert not idx.contains(0)

    # Re-add key 0 — clears tombstone in memory, adds to active shard
    idx.add(0, np.random.rand(32).astype(np.float32))
    assert idx.contains(0)

    # Simulate crash during save: shard write fails
    def crash_on_shard(data, target):
        if str(target).endswith(".usearch"):
            raise OSError("Simulated crash")

    with patch("iscc_usearch.sharded.durable_write", side_effect=crash_on_shard):
        with pytest.raises(OSError):
            idx.save()

    # Because tombstones are persisted AFTER shards, the crash prevents the
    # tombstone removal from reaching disk. On reload, key 0 stays tombstoned.
    idx2 = ShardedIndex(ndim=32, path=tmp_path)
    assert not idx2.contains(0), "Tombstoned key must not reappear after failed shard write"
