---
icon: lucide/layers
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

The `path` directory is created automatically. Shard files are named `shard_000.usearch`,
`shard_001.usearch`, etc.

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

## Limitations

`ShardedNphdIndex` (and `ShardedIndex`) use an **append-only** design. These operations are not
supported:

- `remove()` -- vectors cannot be deleted.
- `copy()` / `clear()` / `reset()` -- would require handling multiple files.
- `join()` / `cluster()` / `pairwise_distance()` -- not applicable to sharded storage.
