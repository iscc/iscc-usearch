---
icon: lucide/rocket
---

# Getting Started

This tutorial walks you through creating your first `iscc-usearch` index, adding vectors, searching
for nearest neighbors, and persisting the index to disk.

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

## Create an index

An `NphdIndex` stores binary bit-vectors up to a given maximum dimension. Create one that supports
vectors up to 256 bits (32 bytes):

```python
from iscc_usearch import NphdIndex

index = NphdIndex(max_dim=256)
```

The `max_dim` parameter sets the upper bound on vector length in bits. All vectors you add must fit
within this limit.

## Add vectors

Vectors are NumPy `uint8` arrays where each byte holds 8 bits of the binary code. Every vector
needs an integer key:

```python
import numpy as np

# Add three 32-bit vectors (4 bytes each)
index.add(1, np.array([255, 128, 64, 32], dtype=np.uint8))
index.add(2, np.array([255, 128, 64, 33], dtype=np.uint8))
index.add(3, np.array([255, 128, 64, 32], dtype=np.uint8))
```

You can also add vectors in batch:

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

Query the index with a vector to find the closest matches:

```python
query = np.array([255, 128, 64, 32], dtype=np.uint8)
matches = index.search(query, count=3)

print(matches.keys)  # Array of matching keys, sorted by distance
print(matches.distances)  # Corresponding NPHD distances [0.0, 1.0]
```

Distances are in the range `[0.0, 1.0]` -- `0.0` means identical, `1.0` means every bit differs.

## Retrieve vectors by key

Look up previously stored vectors:

```python
vector = index.get(1)
print(vector)  # array([255, 128, 64, 32], dtype=uint8)

# Missing keys return None
missing = index.get(999)
print(missing)  # None
```

## Save and reload

Persist the index to a file and restore it later:

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

- **[Variable-Length Vectors](variable-length.md)** -- Mix vectors of different bit-lengths in the
    same index.
- **[Persistence](../howto/persistence.md)** -- Learn about `save()`, `load()`, `view()`, and
    `restore()`.
- **[API Reference](../reference/api.md)** -- Full API documentation.
