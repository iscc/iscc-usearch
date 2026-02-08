---
title: Scaling Up
icon: lucide/trending-up
---

# Scaling up with ShardedNphdIndex

This tutorial builds on the [Getting Started](getting-started.md) and
[Variable-Length Vectors](variable-length.md) guides. You will create a sharded index that
handles large datasets with consistent insert throughput and bounded memory usage.

## When to use ShardedNphdIndex

Use `ShardedNphdIndex` instead of `NphdIndex` when:

- Your dataset exceeds available RAM.
- Insert throughput degrades as the index grows.
- You need persistent, append-only storage with automatic shard rotation.

`ShardedNphdIndex` combines variable-length NPHD support with transparent sharding. The API is
nearly identical to `NphdIndex` -- you add vectors and search without managing shards manually.

## Create a sharded index

```python
import numpy as np
from iscc_usearch import ShardedNphdIndex

index = ShardedNphdIndex(
    max_dim=256,
    path="./my_index",
    shard_size=512 * 1024 * 1024,  # 512 MB per shard
)
```

The `path` directory is created automatically. Shard files appear as the index grows.

## Add mixed-resolution vectors

Just like `NphdIndex`, you can mix 64-bit, 128-bit, and 256-bit vectors:

```python
# 64-bit vectors (8 bytes each)
for i in range(100):
    vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
    index.add(i, vec)

# 128-bit vectors (16 bytes each)
for i in range(100, 200):
    vec = np.random.randint(0, 256, size=16, dtype=np.uint8)
    index.add(i, vec)

# 256-bit vectors (32 bytes each)
for i in range(200, 300):
    vec = np.random.randint(0, 256, size=32, dtype=np.uint8)
    index.add(i, vec)
```

When the active shard exceeds `shard_size`, it is saved to disk and reopened as a read-only
memory-mapped view. A fresh active shard takes its place. This happens automatically.

## Search across all shards

Queries fan out across all shards and results are merged:

```python
query = np.random.randint(0, 256, size=8, dtype=np.uint8)
matches = index.search(query, count=10)

for key, dist in zip(matches.keys, matches.distances):
    print(f"Key {key}: distance = {dist:.4f}")
```

NPHD compares only the common prefix, so a 64-bit query finds nearest neighbors among vectors of
any stored resolution.

## Retrieve vectors

Retrieve the original (unpadded) vector by key:

```python
vec = index.get(0)
print(vec)  # Original 64-bit vector

# Batch retrieval
vecs = index.get([0, 100, 200])
```

The bloom filter provides O(1) rejection of non-existent keys, so lookups stay fast regardless of
shard count.

## Save and reopen

```python
# Save current state (active shard + bloom filter)
index.save()

# Reopen later -- auto-detects existing shards and max_dim
index = ShardedNphdIndex(path="./my_index")

# Verify
matches = index.search(query, count=5)
print(matches.keys)
```

## Inspect the index

```python
print(index.size)  # Total vectors across all shards
print(index.shard_count)  # Number of shard files
print(index.max_dim)  # Maximum bits per vector
```

## Shard directory layout

After adding data and saving, the directory looks like this:

```
my_index/
    shard_000.usearch   # view shard (memory-mapped, read-only)
    shard_001.usearch   # view shard (memory-mapped, read-only)
    shard_002.usearch   # active shard (RAM, read-write)
    bloom.isbf          # bloom filter for key lookups
```

Completed shards are immutable. Only the highest-numbered shard is the active shard.

## Next steps

- **[128-bit UUID keys](../howto/uuid-keys.md)** -- Use `ShardedNphdIndex128` when 64-bit keys are
    not enough.
- **[Sharding how-to](../howto/sharding.md)** -- Shard size tuning and configuration.
- **[Sharding design](../explanation/sharding-design.md)** -- Trade-offs and architecture.
- **[Performance](../explanation/performance.md)** -- Benchmarks and optimization.
- **[Architecture](../explanation/architecture.md)** -- Class hierarchy and data flow.
