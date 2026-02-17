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
- You need append-only storage with transparent shard rotation.

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
```

Completed shards are immutable. The highest-numbered shard is the active shard.

## Add data

The API is the same as `NphdIndex`. Shard rotation happens automatically:

```python
import numpy as np

keys = list(range(1000))
vectors = [np.random.randint(0, 256, size=32, dtype=np.uint8) for _ in range(1000)]

for key, vec in zip(keys, vectors):
    index.add(key, vec)
```

When the active shard exceeds `shard_size`, it is saved to disk and reopened in view mode
(read-only, memory-mapped). A new active shard is then created.

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

## Properties

```python
print(index.size)  # Total vectors across all shards
print(index.shard_count)  # Number of shard files
print(index.max_dim)  # Maximum bits per vector

# Lazy iterators (memory-efficient)
for key in index.keys:
    pass
for vec in index.vectors:
    pass
```

## 128-bit key variants

If your keys exceed 64 bits (e.g., composite `(iscc_id_body, chunk_index)` keys for simprint
indexing), use the 128-bit variants:

- `ShardedIndex128` — same as `ShardedIndex` but with `bytes(16)` keys
- `ShardedNphdIndex128` — same as `ShardedNphdIndex` but with `bytes(16)` keys

The API is identical except that keys are `bytes` of length 16 (single) or `np.dtype('V16')`
arrays (batch) instead of integers. See the [UUID keys how-to](uuid-keys.md) for details.

## Limitations

`ShardedNphdIndex` (and `ShardedIndex`) use an **append-only** design. The following operations
raise `NotImplementedError`:

- `remove()` — vectors cannot be deleted.
- `copy()` / `clear()` — would require handling multiple files.
- `join()` / `cluster()` / `pairwise_distance()` — not applicable to sharded storage.
- `upsert()` — not supported (append-only design requires `remove()`).

!!! warning "Single-process only"

    Running multiple processes against the same index files may corrupt data. See
    [Architecture](../explanation/architecture.md#concurrency-model) for details.

!!! note "Required parameters"

    When creating a new sharded index (no existing shards on disk), `max_dim` is required.
    Omitting it raises `ValueError`. When reopening an existing index, `max_dim` is auto-detected
    from the shard metadata.
