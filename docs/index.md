---
icon: lucide/house
description: Python library extending USearch with sharded HNSW indexes, Normalized Prefix Hamming Distance for variable-length binary vectors, and 128-bit UUID key support.
---

# iscc-usearch

[![Tests](https://github.com/iscc/iscc-usearch/actions/workflows/tests.yml/badge.svg)](https://github.com/iscc/iscc-usearch/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/iscc/iscc-usearch/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iscc/iscc-usearch)

**Larger-than-RAM writable HNSW indexes, and variable-length binary vector search.**

## Introduction

`iscc-usearch` is a Python library that extends [USearch](https://github.com/unum-cloud/usearch) - a
[high-performance](https://github.com/unum-cloud/usearch#performance) HNSW library adopted by
ClickHouse, LangChain, and others - with three independent capabilities:

**Sharded HNSW indexes** (`ShardedIndex`) keep a single active shard in RAM for writes while
completed shards are memory-mapped for reads. Works with any vector type and metric USearch
supports, including user-defined distance functions. Insert throughput stays consistent and memory
stays bounded as the index grows to billions of vectors.

**Normalized Prefix Hamming Distance** (`NphdIndex`, `ShardedNphdIndex`) compares binary vectors
of mixed bit-lengths - a 64-bit query finds nearest neighbors among 256-bit vectors with
comparable distances. Purpose-built for [ISCC](https://iscc.codes) (ISO 24138) content
fingerprints, also applicable to [Matryoshka embeddings](https://arxiv.org/abs/2205.13147),
perceptual hashes, and locality-sensitive hashing.

**128-bit UUID keys** (`ShardedIndex128`, `ShardedNphdIndex128`) extend the key space from 64-bit
integers to 128-bit `bytes(16)` keys. Useful when your identifiers are UUIDs, 128-bit hashes, or
structured multi-part keys that don't fit in a `uint64`.

**Key features:**

- **Bounded memory** - only one shard in RAM at a time, the rest memory-mapped
- **Billions of vectors** - sharded indexes scale well beyond single-machine RAM
- **Incremental writes** - append vectors without rebuilding the index
- **Mixed bit-lengths** - 64-bit and 256-bit vectors coexist in the same index
- **128-bit keys** - `bytes(16)` UUID keys when 64-bit integers are not enough
- **Any distance metric** - user-defined metrics via USearch's plugin system
- **Fast** - inherits USearch's HNSW engine, benchmarked at 10x the throughput of FAISS

## Which index class?

| Class                 | Var-len | Keys    | Shards | Use when...                          |
| --------------------- | :-----: | ------- | :----: | ------------------------------------ |
| `NphdIndex`           |    ✓    | uint64  |   —    | Binary variable-length, fits in RAM  |
| `ShardedIndex`        |    —    | uint64  |   ✓    | Exceeds RAM, any metric              |
| `ShardedIndex128`     |    —    | 128-bit |   ✓    | Same, with 128-bit keys              |
| `ShardedNphdIndex`    |    ✓    | uint64  |   ✓    | Binary variable-length, exceeds RAM  |
| `ShardedNphdIndex128` |    ✓    | 128-bit |   ✓    | Binary variable-length, 128-bit keys |

See the [architecture overview](explanation/architecture.md#choosing-an-index-class) for the full
class hierarchy.

## Quick start

```bash
pip install iscc-usearch
```

=== "Variable-length binary (NphdIndex)"

    ```python
    import numpy as np
    from iscc_usearch import NphdIndex

    index = NphdIndex(max_dim=256)

    # Mix 64-bit and 128-bit vectors in the same index
    index.add(1, np.array([255, 128, 64, 32, 16, 8, 4, 2], dtype=np.uint8))
    index.add(2, np.array([255, 128, 64, 32, 16, 8, 4, 2, 1, 0, 255, 128, 64, 32, 16, 8], dtype=np.uint8))

    # Search with a 64-bit query - NPHD compares the common prefix
    query = np.array([255, 128, 64, 32, 16, 8, 4, 2], dtype=np.uint8)
    matches = index.search(query, count=2)

    print(matches.keys)  # Nearest neighbor keys
    print(matches.distances)  # NPHD distances in [0.0, 1.0]
    ```

=== "Sharded HNSW (ShardedIndex)"

    ```python
    import numpy as np
    from iscc_usearch import ShardedIndex

    # Shards are stored in a directory on disk
    index = ShardedIndex(ndim=64, path="my_index", dtype="f32")

    # Add vectors - shards rotate automatically when size limit is reached
    keys = list(range(1000))
    vectors = np.random.rand(1000, 64).astype(np.float32)
    index.add(keys, vectors)

    # Search across all shards
    matches = index.search(vectors[0], count=10)

    print(matches.keys)  # Nearest neighbor keys
    print(matches.distances)  # Cosine distances
    ```

## Architecture

!!! note "ShardedIndex architecture"

    ![ShardedIndex architecture overview](assets/sharded-index-architecture.avif)

!!! note "NphdIndex architecture"

    ![NphdIndex architecture overview](assets/nphd-index-architecture.avif)

## Documentation

<div class="grid cards" markdown>

- **[Tutorials](tutorials/getting-started.md)** - Learn the basics

    Hands-on guides from installation to working code.

- **[How-to guides](howto/persistence.md)** - Solve specific problems

    Recipes for persistence, sharding, upsert, and bloom filters.

- **[Explanation](explanation/nphd-metric.md)** - Understand the design

    Background on NPHD, architecture, sharding, and performance.

- **[Reference](reference/api.md)** - API details

    Auto-generated API documentation for all public classes.

</div>

[Development & Contributing](development/contributing.md) -
Dev setup, testing, and contribution guidelines.

[Source code on GitHub :lucide-external-link:](https://github.com/iscc/iscc-usearch){ .md-button }
