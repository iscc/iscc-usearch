# iscc-usearch

Scalable approximate nearest neighbor search (ANNS) for variable-length binary bit-vectors.

## Overview

`iscc-usearch` is a thin wrapper around [USearch](https://github.com/unum-cloud/usearch) that adds:

- **Variable-length bit-vectors**: Store and search vectors of different lengths in the same index
- **Normalized Prefix Hamming Distance (NPHD)**: A custom metric for prefix-compatible binary codes
- **Automatic sharding**: Scale beyond single-file limitations with transparent multi-shard indexes

This library is designed for [ISCC](https://iscc.codes) content fingerprinting, where shorter codes are valid prefixes of longer ones.

## Installation

```bash
pip install iscc-usearch
```

## Quick Start

```python
import numpy as np
from iscc_usearch import NphdIndex

# Create an index for up to 256-bit vectors
index = NphdIndex(max_dim=256)

# Add variable-length vectors with integer keys
index.add(1, np.array([255, 128, 64, 32], dtype=np.uint8))  # 32-bit
index.add(2, np.array([255, 128, 64, 33], dtype=np.uint8))  # 32-bit (1 bit different)
index.add(3, np.array([255, 128], dtype=np.uint8))  # 16-bit prefix

# Search returns matches sorted by NPHD
query = np.array([255, 128, 64, 32], dtype=np.uint8)
matches = index.search(query, count=3)

print(matches.keys)  # [1, 2, 3]
print(matches.distances)  # [0.0, 0.03125, 0.0]  # Normalized distances
```

## The NPHD Metric

Standard Hamming distance fails when comparing vectors of different lengths. NPHD solves this by:

1. Comparing only the **common prefix** (the shorter of the two vectors)
1. Counting bit differences in that prefix
1. **Normalizing** by the prefix length (distance / bits)

This produces distances in the range [0.0, 1.0] regardless of vector lengths, enabling meaningful similarity comparisons between codes of different granularity.

## Index Classes

### NphdIndex

Single-file index for variable-length binary vectors with NPHD metric.

```python
from iscc_usearch import NphdIndex

# Create index
index = NphdIndex(max_dim=256)  # Max 256 bits (32 bytes) per vector

# Add vectors (single or batch)
index.add(key, vector)
index.add(keys, vectors)

# Search
matches = index.search(query, count=10)
matches = index.search(queries, count=10)  # Batch search

# Retrieve vectors by key
vector = index.get(key)

# Persistence
index.save("index.usearch")
index = NphdIndex.restore("index.usearch")
```

### ShardedNphdIndex

Multi-shard index for datasets that exceed single-file limitations. Automatically rotates shards when size limits are reached.

```python
from iscc_usearch import ShardedNphdIndex

# Create sharded index (rotates shards at 1GB by default)
index = ShardedNphdIndex(
    max_dim=256,
    path="./index_shards",
    shard_size=1024 * 1024 * 1024,  # 1GB per shard
)

# Same API as NphdIndex
index.add(keys, vectors)
matches = index.search(query, count=10)

# Save current state
index.save()

# Reopen existing index (auto-detects and loads existing shards)
index = ShardedNphdIndex(path="./index_shards")
```

### ShardedIndex

Generic sharded index for any metric (not just NPHD). Use this for standard vector types.

```python
from iscc_usearch import ShardedIndex
from usearch.index import MetricKind

index = ShardedIndex(
    ndim=128,
    metric=MetricKind.Cos,
    path="./shards",
)
```

## Concurrency

All index classes are **single-process only**. The underlying `.usearch` files have no file locking. Running multiple processes against the same index may corrupt data.

For concurrent access, use a single process with async/await patterns.

## API Reference

### NphdIndex

| Method                   | Description                           |
| ------------------------ | ------------------------------------- |
| `add(keys, vectors)`     | Add vectors with integer keys         |
| `upsert(keys, vectors)`  | Insert or update vectors (idempotent) |
| `search(vectors, count)` | Find k nearest neighbors              |
| `get(keys)`              | Retrieve vectors by key               |
| `save(path)`             | Save index to file                    |
| `load(path)`             | Load index from file                  |
| `view(path)`             | Memory-map index (read-only)          |
| `restore(path)`          | Static method to restore from file    |
| `copy()`                 | Create a copy of the index            |

### ShardedNphdIndex / ShardedIndex

| Method                   | Description                            |
| ------------------------ | -------------------------------------- |
| `add(keys, vectors)`     | Add vectors (auto-rotates shards)      |
| `search(vectors, count)` | Search across all shards               |
| `get(keys)`              | Retrieve vectors by key from any shard |
| `contains(keys)`         | Check existence across all shards      |
| `count(keys)`            | Count occurrences across all shards    |
| `save()`                 | Save active shard and bloom filter     |

| Property      | Description                         |
| ------------- | ----------------------------------- |
| `size`        | Total vectors across all shards     |
| `shard_count` | Number of shard files               |
| `keys`        | Lazy iterator over all keys         |
| `vectors`     | Lazy iterator over all vectors      |
| `max_dim`     | Maximum bits per vector (NPHD only) |

## License

Apache-2.0
