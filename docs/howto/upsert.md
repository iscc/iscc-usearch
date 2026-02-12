---
icon: lucide/replace
description: Perform idempotent insert-or-update operations with upsert. Compare upsert vs add for single and batch operations on NphdIndex and ShardedNphdIndex.
---

# Upsert

`upsert()` is an idempotent insert-or-update operation.

## What upsert does

`upsert(keys, vectors)` ensures that each key maps to the given vector:

- **Key is new**: The vector is inserted (same as `add()`).
- **Key exists, vector unchanged**: No operation (skip).
- **Key exists, vector changed**: The old vector is removed and the new vector is inserted.

`upsert()` is idempotent: calling it multiple times with the same inputs produces the same result.

## Single upsert

```python
import numpy as np
from iscc_usearch import NphdIndex

index = NphdIndex(max_dim=256)

vec = np.array([255, 128, 64, 32], dtype=np.uint8)
index.upsert(1, vec)

# Calling again with same data is a no-op
index.upsert(1, vec)
print(len(index))  # 1

# Update with different vector
vec_new = np.array([0, 0, 0, 0], dtype=np.uint8)
index.upsert(1, vec_new)
print(index.get(1))  # array([0, 0, 0, 0], dtype=uint8)
```

## Batch upsert

Batch upsert works with uniform-length vectors:

```python
keys = [1, 2, 3]
vectors = np.array(
    [
        [255, 128, 64, 32],
        [0, 0, 0, 0],
        [1, 2, 3, 4],
    ],
    dtype=np.uint8,
)

index.upsert(keys, vectors)
```

## Variable-length batch upsert

Batch `upsert()` requires all vectors to have the same length because it normalizes inputs to a
2D array internally. For variable-length vectors, call `upsert()` one at a time:

```python
variable_keys = [10, 11, 12]
variable_vecs = [
    np.array([255, 128], dtype=np.uint8),  # 16-bit
    np.array([255, 128, 64, 32], dtype=np.uint8),  # 32-bit
    np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8),  # 64-bit
]

for key, vec in zip(variable_keys, variable_vecs):
    index.upsert(key, vec)
```

## When to use upsert vs add

| Scenario                          | Use        |
| --------------------------------- | ---------- |
| Keys are guaranteed unique        | `add()`    |
| Keys may repeat (idempotent sync) | `upsert()` |
| Bulk initial load                 | `add()`    |
| Incremental updates               | `upsert()` |

!!! note

    `upsert()` requires explicit keys. Passing `keys=None` raises `ValueError`.

!!! note

    `upsert()` is available on single-file indexes only (`NphdIndex`, `Index`), including
    uuid-keyed indexes. Sharded indexes do not support `upsert()` because they use an
    append-only design without `remove()`.

    The number of keys and vectors must match, or `ValueError` is raised.
