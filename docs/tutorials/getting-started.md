---
icon: lucide/rocket
---

# Getting started

Build your first `iscc-usearch` index, add vectors, search for nearest neighbors, and save the
index to disk.

## Prerequisites

- Python 3.12 or later
- `pip` or [`uv`](https://docs.astral.sh/uv/)

## Install

```bash
pip install iscc-usearch
```

Or with `uv`:

```bash
uv add iscc-usearch
```

### Verify the installation

```python
import iscc_usearch

print(iscc_usearch.__version__)
```

### What gets installed

`iscc-usearch` brings in four runtime dependencies:

| Dependency                                               | Purpose                                                                            |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| [`usearch-iscc`](https://pypi.org/project/usearch-iscc/) | Patched USearch fork with fast `view()` and GIL release for parallel shard loading |
| [`numba`](https://numba.pydata.org/)                     | JIT compilation for the NPHD metric and vector padding functions                   |
| [`fastbloom-rs`](https://pypi.org/project/fastbloom-rs/) | Rust-based bloom filter for O(1) key rejection in sharded indexes                  |
| [`loguru`](https://loguru.readthedocs.io/)               | Structured logging                                                                 |

## Create an index

`NphdIndex` stores binary bit-vectors up to a given maximum dimension. Here we create one that
accepts vectors up to 256 bits (32 bytes):

```python
from iscc_usearch import NphdIndex

index = NphdIndex(max_dim=256)
```

`max_dim` is the upper bound on vector length in bits. Every vector you add must fit within this
limit.

!!! note "Constraints on `max_dim`"

    `max_dim` must be a multiple of 8 and at most 256 (the maximum resolution of ISCC content
    fingerprints). The constructor raises `ValueError` if either constraint is violated.

## Add vectors

Vectors are NumPy `uint8` arrays where each byte holds 8 bits of the binary code. Each vector
requires an integer key:

```python
import numpy as np

# Add three 32-bit vectors (4 bytes each)
index.add(1, np.array([255, 128, 64, 32], dtype=np.uint8))
index.add(2, np.array([255, 128, 64, 33], dtype=np.uint8))
index.add(3, np.array([255, 128, 64, 32], dtype=np.uint8))
```

Batch insertion works too:

```python
keys = [10, 11, 12]
vectors = np.array(
    [
        [255, 128, 64, 32],
        [255, 128, 64, 33],
        [0, 0, 0, 0],
    ],
    dtype=np.uint8,
)

index.add(keys, vectors)
```

## Search for nearest neighbors

Pass a query vector to find the closest matches:

```python
query = np.array([255, 128, 64, 32], dtype=np.uint8)
matches = index.search(query, count=3)

print(matches.keys)  # Array of matching keys, sorted by distance
print(matches.distances)  # Corresponding NPHD distances [0.0, 1.0]
```

Distances range from `0.0` (identical) to `1.0` (every bit differs).

## Retrieve vectors by key

Fetch a stored vector by its key:

```python
vector = index.get(1)
print(vector)  # array([255, 128, 64, 32], dtype=uint8)

# Missing keys return None
missing = index.get(999)
print(missing)  # None
```

## Save and reload

Save the index to a file and load it back later:

```python
# Save
index.save("my_index.usearch")

# Restore (loads into RAM)
restored = NphdIndex.restore("my_index.usearch")

# Verify it works
matches = restored.search(query, count=3)
print(matches.keys)
```

!!! tip

    For read-only access with lower memory usage, use `restore(..., view=True)` to memory-map the
    file instead of loading it fully into RAM. See the
    [Persistence how-to](../howto/persistence.md) for details.

## Next steps

- **[Variable-length vectors](variable-length.md)** -- Mix vectors of different bit-lengths in the
    same index.
- **[Scaling up](scaling-up.md)** -- Use `ShardedNphdIndex` for datasets that exceed RAM.
- **[Persistence](../howto/persistence.md)** -- `save()`, `load()`, `view()`, and `restore()`
    explained.
- **[API reference](../reference/api.md)** -- Full API documentation.
