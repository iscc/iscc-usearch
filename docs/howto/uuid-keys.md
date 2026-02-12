---
icon: lucide/key-round
description: Use ShardedIndex128 and ShardedNphdIndex128 for datasets with 128-bit identifiers such as UUIDs, MD5 hashes, or structured multi-part keys.
---

# 128-bit UUID keys

Use `ShardedIndex128` or `ShardedNphdIndex128` when 64-bit integer keys are not enough — for
example, when your identifiers are UUIDs, 128-bit hashes, or structured multi-part keys.

## When to use 128-bit keys

Switch from `ShardedIndex` / `ShardedNphdIndex` to their 128-bit variants when:

- Your key space exceeds 64 bits.
- Your identifiers are natively 128-bit (UUIDs, MD5 hashes, etc.).
- You need to pack multiple fields into a single key (e.g., two 8-byte values).

## Key format

128-bit keys are represented as:

- **Single key**: `bytes` of length 16
- **Batch keys**: NumPy array with `dtype='V16'` (void 16-byte elements)

```python
import numpy as np

# Single key — 16 bytes
key = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"

# Batch keys — V16 array
keys = np.array([key, b"\xff" * 16], dtype="V16")
```

## Create a 128-bit sharded index

=== "Fixed-length vectors (ShardedIndex128)"

    ```python
    import numpy as np
    from iscc_usearch import ShardedIndex128

    index = ShardedIndex128(
        ndim=256,
        path="./my_index_128",
        dtype="f32",
        shard_size=512 * 1024 * 1024,
    )
    ```

=== "Variable-length NPHD (ShardedNphdIndex128)"

    ```python
    import numpy as np
    from iscc_usearch import ShardedNphdIndex128

    index = ShardedNphdIndex128(
        max_dim=256,
        path="./my_nphd_128",
        shard_size=512 * 1024 * 1024,
    )
    ```

## Add vectors

```python
import numpy as np

# Single add
key = b"\x00" * 16
vector = np.random.randint(0, 256, size=32, dtype=np.uint8)
index.add(key, vector)

# Batch add
keys = np.array([b"\x00" * 15 + bytes([i]) for i in range(100)], dtype="V16")
vectors = np.random.randint(0, 256, size=(100, 32), dtype=np.uint8)
index.add(keys, vectors)
```

!!! note

    Auto-key generation (`keys=None`) is not supported for 128-bit indexes. All keys must be
    provided explicitly.

## Search

Search works the same as with 64-bit indexes. Results contain `V16` keys:

```python
query = np.random.randint(0, 256, size=32, dtype=np.uint8)
matches = index.search(query, count=10)

for key_bytes, dist in zip(matches.keys, matches.distances):
    print(f"Key {bytes(key_bytes).hex()}: distance = {dist:.4f}")
```

## Retrieve by key

```python
# Single get
vector = index.get(b"\x00" * 16)

# Batch get
keys = np.array([b"\x00" * 16, b"\xff" * 16], dtype="V16")
vectors = index.get(keys)

# Contains
print(index.contains(b"\x00" * 16))  # True or False
print(b"\x00" * 16 in index)  # Same thing
```

## Structured key packing

A common pattern is packing two 8-byte values into a 16-byte key. Use `struct` for this:

```python
import struct

part_a = 0x0123456789ABCDEF  # first 8 bytes
part_b = 42  # second 8 bytes

# Pack as big-endian: part_a (8B) + part_b (8B)
key = struct.pack(">QQ", part_a, part_b)

# Add to index
index.add(key, vector)

# Later, unpack from search results
a, b = struct.unpack(">QQ", key)
```

## Save and reopen

```python
# Save
index.save()

# Reopen — auto-detects existing shards and uuid key kind
index = ShardedIndex128(path="./my_index_128")
# or
index = ShardedNphdIndex128(path="./my_nphd_128")
```

## Validation rules

128-bit indexes enforce strict key validation:

| Operation                                      | Validation                                        |
| ---------------------------------------------- | ------------------------------------------------- |
| `add(key, vec)`                                | `key` must be `bytes` of length 16                |
| `add(keys, vecs)`                              | `keys` must be `np.ndarray` with `dtype='V16'`    |
| `get(key)` / `contains(key)` / `count(key)`    | `key` must be `bytes` of length 16                |
| `get(keys)` / `contains(keys)` / `count(keys)` | `keys` must be V16 array or `Sequence[bytes(16)]` |

Passing the wrong key type or length raises `ValueError` immediately rather than producing
silent incorrect results.

## Limitations

- **No auto-keys**: `keys=None` raises `ValueError`. All keys must be explicit.
- **Append-only**: Same as standard sharded indexes — no `remove()`, `copy()`, or `clear()`.
    `upsert()` is available on the single-file `Index` (including uuid keys) but not on sharded
    indexes.
