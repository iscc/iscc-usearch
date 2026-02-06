# iscc-usearch

[![Tests](https://github.com/iscc/iscc-usearch/actions/workflows/tests.yml/badge.svg)](https://github.com/iscc/iscc-usearch/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

A thin wrapper around [USearch](https://github.com/unum-cloud/usearch) providing scalable
approximate nearest neighbor search (ANNS) for variable-length binary bit-vectors. Designed for
[ISCC](https://iscc.codes) content fingerprinting where shorter codes are valid prefixes of longer
ones, searchable in the same index using the Normalized Prefix Hamming Distance (NPHD) metric.

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
print(matches.distances)  # [0.0, 0.03125, 0.0]
```

## Documentation

Full documentation: **https://iscc.github.io/iscc-usearch/**

- [Tutorials](https://iscc.github.io/iscc-usearch/tutorials/) -- Step-by-step getting started guides
- [How-to Guides](https://iscc.github.io/iscc-usearch/howto/) -- Persistence, sharding, upsert, bloom filters
- [Explanation](https://iscc.github.io/iscc-usearch/explanation/) -- NPHD metric, architecture, performance
- [API Reference](https://iscc.github.io/iscc-usearch/reference/api/) -- Auto-generated from source

## License

Apache-2.0
