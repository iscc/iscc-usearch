---
icon: lucide/blocks
---

# Architecture

## The core problem

USearch requires all vectors in an index to have the same dimensionality. ISCC codes are
variable-length (64-bit to 256-bit). `iscc-usearch` solves this with **length-prefixed padding**
and a **custom metric** that ignores padding during distance computation.

## Length-prefixed padding

![NphdIndex architecture overview](../assets/nphd-index-architecture.avif)

Every vector is padded to a uniform size before storage. The first byte holds the original vector
length (in bytes), then the vector data, then zero-padding.

The padded vector is stored in USearch as a `ScalarKind.B1` (binary) vector with
`ndim = max_dim + 8` bits (the extra 8 bits account for the length byte).

On retrieval, `unpad_vectors` reads the length byte and returns only the valid data bytes.

Both `pad_vectors` and `unpad_vectors` are compiled with Numba `@njit(cache=True)` for native speed.

## Custom metric restoration

USearch serializes the metric *kind* (e.g., Hamming) but not the custom function pointer. When an
index is loaded or viewed, USearch substitutes the standard metric for that kind. Since NPHD is
registered as `MetricKind.Hamming` (the closest built-in kind), a loaded index would use standard
Hamming distance instead of NPHD.

`NphdIndex.load()` and `NphdIndex.view()` call `change_metric()` after every load or view to
restore the NPHD function pointer. This happens automatically and callers never need to do it
manually.

## Numba compilation strategy

Two compilation modes are used:

- **`@cfunc`** for the NPHD metric: produces a C-callable function pointer compatible with
    USearch's `CompiledMetric` interface. USearch calls it from C++ during graph traversal without
    crossing the Python/C boundary.

- **`@njit(cache=True)`** for padding functions: JIT-compiled with result caching so recompilation
    cost is paid only once per environment. Operates on NumPy arrays.

## Maximum vector size

The NPHD metric is compiled with a fixed buffer size of 33 bytes (1 length byte + 32 data bytes),
so `max_dim` is capped at **256 bits**. This matches the maximum resolution of ISCC content
fingerprints. `NphdIndex` validates this at construction time.

## Index class hierarchy

`iscc-usearch` provides six index classes. Two are single-file, four are sharded:

```
Index                            (usearch wrapper, uint64 keys)
└── NphdIndex                    (variable-length + NPHD metric)

ShardedIndex                     (composition-based sharding, uint64 keys)
├── ShardedIndex128              (128-bit UUID keys via _UuidKeyMixin)
└── ShardedNphdIndex             (variable-length + NPHD metric)
    └── ShardedNphdIndex128      (128-bit UUID keys via _UuidKeyMixin)
```

`NphdIndex` inherits from `Index` (which extends USearch's `Index`) and adds padding and the NPHD
metric. `ShardedIndex` is a standalone composition-based class that manages multiple USearch indexes
as shards. `ShardedNphdIndex` extends `ShardedIndex` with variable-length vector support and NPHD.

The `128` variants add 128-bit key support using usearch's `key_kind="uuid"` mode. Keys are
`bytes(16)` for single operations and `np.dtype('V16')` arrays for batches. All 128-bit logic
concentrates in a `_UuidKeyMixin` that overrides key-handling hooks on `ShardedIndex` — the base
classes stay clean.

### Choosing an index class

| Class                 | Var-len | Keys    | Shards | Upsert | Use case                              |
| --------------------- | :-----: | ------- | :----: | :----: | ------------------------------------- |
| `Index`               |    —    | uint64  |   —    |   ✓    | Fixed-length vectors, small datasets  |
| `NphdIndex`           |    ✓    | uint64  |   —    |   ✓    | ISCC codes, fits in RAM               |
| `ShardedIndex`        |    —    | uint64  |   ✓    |   —    | Fixed-length vectors, large scale     |
| `ShardedIndex128`     |    —    | 128-bit |   ✓    |   —    | Fixed-length vectors, 128-bit keys    |
| `ShardedNphdIndex`    |    ✓    | uint64  |   ✓    |   —    | ISCC codes, large scale (production)  |
| `ShardedNphdIndex128` |    ✓    | 128-bit |   ✓    |   —    | ISCC codes, large scale, 128-bit keys |

For most ISCC workloads, use **`NphdIndex`** for datasets that fit in RAM, or
**`ShardedNphdIndex`** for datasets that exceed RAM or need consistent insert throughput. Use the
`128` variants when you need keys beyond 64 bits — see the
[UUID keys how-to](../howto/uuid-keys.md).

## Data flow

### Write path (add)

For `NphdIndex`, the application calls `add(key, vector)`, which pads the vector (prepends a length
byte, zero-fills to `max_dim + 8` bits) via `pad_vectors` and stores it in the USearch HNSW graph.

For `ShardedNphdIndex`, the write path adds bloom filter updates and automatic shard rotation: once
the active shard exceeds `shard_size`, it is saved to disk and reopened as a memory-mapped view
while a fresh active shard is created.

### Read path (search)

The query is padded and searched across all shards in parallel (active shard in RAM plus
memory-mapped view shards). Each shard invokes the NPHD metric for distance computations, returning
distances in [0.0, 1.0]. Results are merged via argsort and top-k selection.

## Concurrency model

`iscc-usearch` is designed for **single-process** access. The underlying `.usearch` files have no
file locking or multi-process coordination.

!!! warning

    Running multiple processes against the same index files may corrupt data.

Within a single process, use `async`/`await` for concurrent connections (e.g., serving search
queries from an async web framework). The index objects themselves are not thread-safe -- guard
concurrent access with a lock if using threads.

For sharded indexes, shard rotation (save + reopen) is not atomic. Concurrent reads during rotation
are safe because completed view shards are immutable, but concurrent writes must be serialized.

## Why a thin wrapper

`iscc-usearch` does not fork USearch's index logic. It wraps the existing `Index` class, adds
padding at the boundary, and restores the metric after persistence operations. This keeps the
wrapper small and lets it track upstream USearch improvements without merge conflicts.

The one exception is the [patched usearch fork](performance.md), which modifies USearch's C++ core
for performance. Those patches are confined to view/load paths and don't change the index format.
