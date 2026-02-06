# iscc-usearch

[![Tests](https://github.com/iscc/iscc-usearch/actions/workflows/tests.yml/badge.svg)](https://github.com/iscc/iscc-usearch/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iscc/iscc-usearch)

**Scalable approximate nearest neighbor search for variable-length binary bit-vectors.**

`iscc-usearch` extends [USearch](https://github.com/unum-cloud/usearch) with capabilities
purpose-built for [ISCC](https://iscc.codes) (ISO 24138) content fingerprints: indexing
binary vectors of mixed bit-lengths in a single index, and scaling beyond available RAM through
transparent sharding.

## Why not plain USearch?

USearch is a fast, general-purpose vector index -- but it assumes all vectors have the same
dimensionality, and a single index must fit in memory for writes. ISCC codes break both
assumptions:

- **Variable-length codes.** An ISCC content fingerprint can be 64, 128, or 256 bits depending
    on resolution. Shorter codes are prefixes of longer ones -- a design shared with
    [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147). A useful index must
    store and compare all resolutions together.

- **Large-scale collections.** Real-world content registries grow to hundreds of millions of
    fingerprints. Write throughput in HNSW graphs degrades as the graph grows, and the full graph
    must be loaded into RAM for inserts.

`iscc-usearch` solves both problems with two core additions:

**Normalized Prefix Hamming Distance (NPHD)** compares only the bits that both vectors share and
normalizes the result to `[0.0, 1.0]`. A 64-bit query can find its nearest neighbors among
256-bit vectors -- distances remain comparable across resolutions.

**Transparent sharding** keeps a single active shard in RAM for writes while completed shards are
memory-mapped for reads. This maintains consistent insert throughput regardless of index size and
keeps the memory footprint bounded.

## Installation

```bash
pip install iscc-usearch
```

## Quick Start

```python
import numpy as np
from iscc_usearch import NphdIndex

index = NphdIndex(max_dim=256)

# Mix 64-bit and 128-bit vectors in the same index
index.add(1, np.array([255, 128, 64, 32, 16, 8, 4, 2], dtype=np.uint8))
index.add(2, np.array([255, 128, 64, 32, 16, 8, 4, 2, 1, 0, 255, 128, 64, 32, 16, 8], dtype=np.uint8))

# Search with a 64-bit query -- NPHD compares the common prefix
query = np.array([255, 128, 64, 32, 16, 8, 4, 2], dtype=np.uint8)
matches = index.search(query, count=2)

print(matches.keys)  # Nearest neighbor keys
print(matches.distances)  # NPHD distances in [0.0, 1.0]
```

## Documentation

Full documentation: **https://iscc.github.io/iscc-usearch/**

- [Tutorials](https://iscc.github.io/iscc-usearch/tutorials/) -- Step-by-step getting started guides
- [How-to Guides](https://iscc.github.io/iscc-usearch/howto/) -- Persistence, sharding, upsert, bloom filters
- [Explanation](https://iscc.github.io/iscc-usearch/explanation/) -- NPHD metric, architecture, performance
- [API Reference](https://iscc.github.io/iscc-usearch/reference/api/) -- Auto-generated from source
- [Development](https://iscc.github.io/iscc-usearch/development/) -- Dev setup, testing, and contribution guidelines

## License

Apache-2.0
