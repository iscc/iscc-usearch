---
title: Variable-Length
icon: lucide/ruler
---

# Variable-length vectors

This tutorial builds on the [Getting Started](getting-started.md) guide. You will add vectors of
different bit-lengths to the same index and see how NPHD handles prefix matching.

## The prefix relationship

In ISCC content fingerprinting, a shorter code is always a prefix of a longer one. A 64-bit code
holds the most significant bits. A 128-bit code extends it with finer detail. A 256-bit code is
the full-resolution fingerprint.

```mermaid
graph LR
    A["64-bit<br/>(8 bytes)"] -->|prefix of| B["128-bit<br/>(16 bytes)"]
    B -->|prefix of| C["256-bit<br/>(32 bytes)"]
    style A fill:#e8f4fd,stroke:#1976d2
    style B fill:#e8f4fd,stroke:#1976d2
    style C fill:#e8f4fd,stroke:#1976d2
```

`iscc-usearch` stores all these lengths in a single index and compares them with the
Normalized Prefix Hamming Distance (NPHD).

## Create an index with mixed-length vectors

```python
import numpy as np
from iscc_usearch import NphdIndex

index = NphdIndex(max_dim=256)

# 64-bit vector (8 bytes)
v64 = np.array([255, 128, 64, 32, 16, 8, 4, 2], dtype=np.uint8)

# 128-bit vector (16 bytes) -- first 8 bytes match v64
v128 = np.array([255, 128, 64, 32, 16, 8, 4, 2, 1, 0, 255, 128, 64, 32, 16, 8], dtype=np.uint8)

# 256-bit vector (32 bytes) -- first 16 bytes match v128
v256 = np.array(
    [
        255,
        128,
        64,
        32,
        16,
        8,
        4,
        2,
        1,
        0,
        255,
        128,
        64,
        32,
        16,
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
        0,
        255,
        254,
        253,
        252,
        251,
        250,
        249,
        248,
    ],
    dtype=np.uint8,
)

index.add(1, v64)
index.add(2, v128)
index.add(3, v256)
```

## Search with a short query

A 64-bit query causes NPHD to compare only the first 64 bits of each stored vector:

```python
query = v64.copy()
matches = index.search(query, count=3)

for key, dist in zip(matches.keys, matches.distances):
    print(f"Key {key}: distance = {dist:.4f}")
```

Expected output:

```
Key 1: distance = 0.0000
Key 2: distance = 0.0000
Key 3: distance = 0.0000
```

All three vectors share the same first 8 bytes, so every distance is `0.0`.

## Search with a longer query

Now search with the 128-bit vector. NPHD compares 128 bits against vectors that are at least that
long, but only 64 bits against the shorter vector:

```python
query = v128.copy()
matches = index.search(query, count=3)

for key, dist in zip(matches.keys, matches.distances):
    print(f"Key {key}: distance = {dist:.4f}")
```

The 128-bit and 256-bit vectors match over all 128 compared bits. The 64-bit vector still
matches over its shorter prefix.

## How NPHD distances work

NPHD divides the Hamming distance by the length of the shorter vector:

```
NPHD(a, b) = hamming(prefix_a, prefix_b) / min(bits_a, bits_b)
```

Properties:

- **Range**: Always `[0.0, 1.0]`, regardless of vector lengths.
- **Prefix compatibility**: A 64-bit vector matching the first 64 bits of a 256-bit vector has
    distance `0.0`.
- **Symmetry**: `NPHD(a, b) == NPHD(b, a)`.

For the math behind this, see the [NPHD metric explanation](../explanation/nphd-metric.md).

## Introduce a difference

Add a vector that differs by one bit in the first byte:

```python
v64_diff = np.array([254, 128, 64, 32, 16, 8, 4, 2], dtype=np.uint8)  # 254 vs 255
index.add(4, v64_diff)

matches = index.search(v64, count=4)
for key, dist in zip(matches.keys, matches.distances):
    print(f"Key {key}: distance = {dist:.6f}")
```

The distance between `v64` and `v64_diff` is `1/64 = 0.015625` -- one differing bit out of 64.

## Next steps

- **[NPHD metric](../explanation/nphd-metric.md)** -- Mathematical properties of the distance
    function.
- **[Architecture](../explanation/architecture.md)** -- How variable-length vectors are stored
    internally.
