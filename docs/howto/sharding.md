---
icon: lucide/layers
description: Configure ShardedNphdIndex shard sizes for write-heavy, read-heavy, or balanced workloads. Tune shard rotation and cross-shard search performance.
---

# Sharding

Use `ShardedNphdIndex` when your dataset needs to scale beyond a single index file.

## When to use sharding

Switch to `ShardedNphdIndex` when:

- Your dataset exceeds available RAM.
- Insert throughput degrades as the index grows because HNSW graph construction slows with size.
- You need persistent storage with transparent shard rotation.

## Create a sharded index

```python
from iscc_usearch import ShardedNphdIndex

index = ShardedNphdIndex(
    max_dim=256,
    path="./my_shards",
    shard_size=512 * 1024 * 1024,  # 512 MB per shard
)
```

The `path` directory is created automatically. After adding data and saving, the directory looks
like this:

```
my_shards/
    shard_000.usearch   # view shard (memory-mapped, read-only)
    shard_001.usearch   # view shard (memory-mapped, read-only)
    shard_002.usearch   # active shard (RAM, read-write)
    bloom.isbf          # bloom filter state
    tombstones.npy      # deletion/dedup state (if any removals or upserts pending)
```

Completed shards are read-only. The highest-numbered shard is the active shard.

## Add data

The API is the same as `NphdIndex`. Shard rotation happens automatically:

```python
import numpy as np

keys = list(range(1000))
vectors = [np.random.randint(0, 256, size=32, dtype=np.uint8) for _ in range(1000)]

for key, vec in zip(keys, vectors):
    index.add(key, vec)
```

Duplicate keys within the active shard are silently skipped. Keys already saved in view shards are
not checked by `add()`; use `add_once()` when loading data that may repeat across shard boundaries.

When the active shard exceeds `shard_size`, it is saved to disk and reopened in view mode
(read-only, memory-mapped). A new active shard is then created.

## Background shard rotation

By default, shard rotation blocks `add()` until the old shard is serialized, written to disk,
and reopened as a view. Enable background rotation to move serialization off the hot path:

```python
index = ShardedNphdIndex(
    max_dim=256,
    path="./my_shards",
    shard_size=512 * 1024 * 1024,
    background_rotation=True,
    max_pending_rotations=2,  # backpressure limit (default)
)
```

With `background_rotation=True`, shard rotation detaches the full shard, submits serialization
to a background thread, and creates a new active shard immediately. `add()` returns without
waiting for the disk write to finish.

Rotated data is not searchable until the background task completes and the shard is registered
as a view. Read methods (`search()`, `get()`, `contains()`) do **not** register completed
rotations — call `drain_rotations()` before reads that require complete visibility:

```python
index.drain_rotations()
```

Key-dependent operations (`upsert`, `add_once`, `remove`, `save`, `compact`, `reset`) drain
automatically before executing. `add()` registers completed rotations opportunistically but
does not block.

If `max_pending_rotations` pending tasks queue up, `add()` blocks until the oldest rotation
finishes (backpressure).

## Search across shards

Queries run across all shards automatically:

```python
query = np.random.randint(0, 256, size=32, dtype=np.uint8)
matches = index.search(query, count=10)
print(matches.keys, matches.distances)
```

Results from all shards are merged and sorted by distance.

## Save and reopen

```python
# Save current state (active shard + bloom filter)
index.save()

# Reopen later -- auto-detects existing shards and max_dim
index = ShardedNphdIndex(path="./my_shards")
```

## Close and context manager

`close()` drains pending rotations, saves unsaved state, shuts down the background executor,
and releases memory-mapped resources. Write operations raise `RuntimeError` after close:

```python
index = ShardedNphdIndex(max_dim=256, path="./my_shards", background_rotation=True)
for key, vec in data:
    index.add(key, vec)
index.close()

# index.add(...)  # RuntimeError: index is closed
```

Use the context manager to close automatically on exit:

```python
with ShardedNphdIndex(max_dim=256, path="./my_shards", background_rotation=True) as index:
    for key, vec in data:
        index.add(key, vec)
# index.close() called automatically
```

`close()` is idempotent — safe to call multiple times.

## Read-only mode

Open an existing index for read-only access with `read_only=True`. All shards are memory-mapped
(no active shard in RAM), and write operations raise `RuntimeError`:

```python
index = ShardedNphdIndex(path="./my_shards", read_only=True)

# Search and retrieve work normally
matches = index.search(query, count=10)
vec = index.get(42)

# Writes are blocked
index.add(99, vec)  # RuntimeError: index is read-only
```

Read-only mode requires existing shards on disk — passing `read_only=True` to an empty path raises
`ValueError`.

Use read-only mode when:

- Serving search queries from a pre-built index without risk of accidental writes.
- Running multiple read-only instances against the same shard directory (each process opens its own
    memory-mapped views).

## Skip-if-exists with add_once()

`add_once()` adds vectors only when their keys do not already exist. Existing keys are silently
skipped (first-write-wins):

```python
import numpy as np

vec_a = np.random.randint(0, 256, size=32, dtype=np.uint8)
vec_b = np.random.randint(0, 256, size=32, dtype=np.uint8)

# First add succeeds
index.add_once(1, vec_a)

# Second add is silently skipped — vec_a is kept
index.add_once(1, vec_b)
assert np.array_equal(index.get(1), vec_a)
```

Batch `add_once()` deduplicates within the batch (first occurrence wins) and skips keys already
in the index:

```python
keys = [10, 11, 10]  # duplicate key 10
vecs = np.random.randint(0, 256, size=(3, 32), dtype=np.uint8)
index.add_once(keys, vecs)  # Adds keys 10 and 11 (second key=10 skipped)
```

!!! note

    `add_once()` requires explicit keys — `keys=None` raises `ValueError`.

## Reset the index

`reset()` releases all resources (view shards, active shard, bloom filter) without deleting files
on disk. After reset, the index is empty and ready for new `add()` calls:

```python
print(index.size)  # e.g. 1000
index.reset()
print(index.size)  # 0

# Add fresh data with the same configuration
index.add(1, vec_a)
```

!!! note

    `reset()` does not delete shard files. Call it when you want to release memory and start
    fresh in-process without removing the persisted data.

## Choosing `shard_size`

| Workload    | Recommended shard size | Rationale                              |
| ----------- | ---------------------- | -------------------------------------- |
| Write-heavy | 1/8 of available RAM   | More shards, consistent add throughput |
| Read-heavy  | 1/2 of available RAM   | Fewer shards, lower query latency      |
| Balanced    | 1/4 of available RAM   | Default recommendation                 |

The default is 1 GB. Smaller shards keep insert throughput high but increase query latency because
more shards need to be searched. See [Sharding design](../explanation/sharding-design.md) for
trade-off details.

## Track unsaved changes with `dirty`

The `dirty` property counts unsaved key mutations (adds and removes). Use it to implement
caller-driven flush policies:

```python
flush_threshold = 1000

for i, vec in enumerate(vectors):
    index.add(i, vec)
    if index.dirty >= flush_threshold:
        index.save()  # resets dirty to 0
        print(f"Flushed at {i + 1} vectors")

print(index.dirty)  # Mutations since last save
```

`dirty` resets to 0 on `save()` and `reset()`. Shard rotation does not reset it — unsaved
bloom filter and tombstone state may still need flushing. Read-only indexes always return 0.

## Properties

```python
print(index.size)  # Logical vector count (excludes tombstoned entries)
print(index.shard_count)  # Number of shard files
print(index.max_dim)  # Maximum bits per vector
print(index.dirty)  # Unsaved key mutations since last save

# Lazy iterators (memory-efficient)
for key in index.keys:
    pass
for vec in index.vectors:
    pass
```

Use `stats()` when exporting monitoring or diagnostics:

```python
stats = index.stats()
print(stats["total_vectors"])
print(stats["view_shards"])
print(stats["active_shard_vectors"])
print(stats["dirty"])
print(stats["memory_usage"])
```

The returned dictionary also includes `dimensions`, `metric`, `dtype`, `connectivity`,
`shard_size`, `tombstones`, `bloom_filter`, `path`, and `read_only`.

## 128-bit key variants

If your keys exceed 64 bits (e.g., composite `(iscc_id_body, chunk_index)` keys for simprint
indexing), use the 128-bit variants:

- `ShardedIndex128` — same as `ShardedIndex` but with `bytes(16)` keys
- `ShardedNphdIndex128` — same as `ShardedNphdIndex` but with `bytes(16)` keys

The API is identical except that keys are `bytes` of length 16 (single) or `np.dtype('V16')`
arrays (batch) instead of integers. See the [UUID keys how-to](uuid-keys.md) for details.

## Remove vectors

`remove()` deletes vectors by key. Active shard entries are removed immediately via USearch's
lazy deletion. View shard entries are tombstoned — suppressed on reads and cleaned on
`compact()`:

```python
# Single remove
index.remove(42)

# Batch remove
index.remove([10, 11, 12])

# Python del syntax
del index[42]
```

Keys not found in the index are silently ignored.

!!! note

    `remove()` does not support multi-label indexes (`multi=True`). Passing `multi=True` raises
    `ValueError`.

## Upsert (insert-or-update)

`upsert()` ensures each key maps to the given vector:

- **Key is new**: inserts the vector.
- **Key exists**: removes the old entry and inserts the new vector.

```python
import numpy as np

vec = np.array([255, 128, 64, 32], dtype=np.uint8)
index.upsert(1, vec)

# Update with different vector
vec_new = np.array([0, 0, 0, 0], dtype=np.uint8)
index.upsert(1, vec_new)
print(index.get(1))  # array([0, 0, 0, 0], dtype=uint8)
```

Batch upsert deduplicates within the batch (last occurrence wins):

```python
keys = [1, 2, 1]  # duplicate key 1
vecs = np.random.randint(0, 256, size=(3, 8), dtype=np.uint8)
index.upsert(keys, vecs)  # key 1 gets the third vector
```

!!! note

    `upsert()` does not support multi-label indexes (`multi=True`). Passing `multi=True` raises
    `ValueError`.

See the [Upsert how-to](upsert.md) for details on `upsert()` vs `add()` vs `add_once()`.

## Compact the index

After removing or upserting vectors, tombstoned entries still occupy space in view shard files.
`compact()` rebuilds view shards to reclaim that space:

```python
removed = index.compact()
print(f"Compaction removed {removed} entries")
```

Compaction processes shards newest-to-oldest, dropping tombstoned entries and cross-shard
duplicates. It then saves the rebuilt index (including an updated bloom filter).

!!! tip

    Compaction is optional. Tombstoned entries are already filtered from search results and
    iterators. Compact when disk space matters or tombstone density is high.

## Shard directory layout

After adding, removing, and saving data, the directory looks like this:

```
my_shards/
    shard_000.usearch     # view shard (memory-mapped, read-only)
    shard_001.usearch     # view shard (memory-mapped, read-only)
    shard_002.usearch     # active shard (RAM, read-write)
    bloom.isbf            # bloom filter state
    tombstones.npy        # deletion/dedup state (present after remove or upsert)
```

The `tombstones.npy` file persists tombstoned keys and signals that cross-shard deduplication
filtering is needed. It is created on first `remove()` or `upsert()` and cleared after
`compact()`.

## Limitations

The following operations raise `NotImplementedError`:

- `copy()` / `clear()` — would require handling multiple files.
- `join()` / `cluster()` / `pairwise_distance()` — not applicable to sharded storage.
- `rename()` — not supported.

!!! warning "Single-process only"

    Running multiple processes against the same index files may corrupt data. See
    [Architecture](../explanation/architecture.md#concurrency-model) for details.

!!! note "Required parameters"

    When creating a new sharded index (no existing shards on disk), `max_dim` is required.
    Omitting it raises `ValueError`. When reopening an existing index, `max_dim` is auto-detected
    from the shard metadata.
