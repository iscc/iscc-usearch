# Performance Optimizations

`iscc-usearch` depends on a [patched usearch fork](https://github.com/iscc/usearch)
(v2.23.2) that includes targeted patches for ISCC-specific workloads. usearch is an
excellent high-performance ANNS library; our fork adds specialized behavior for the
heavy sharding, read-mostly access patterns, and large-scale memory-mapped indexes
that ISCC indexing requires. This page documents the optimizations in both the fork
and in `iscc-usearch` itself.

## Patched usearch Fork

Pre-built wheels are hosted at <https://iscc.github.io/usearch/> for Python 3.12+
on macOS, Linux (x86_64, aarch64), and Windows (AMD64, ARM64).

### Instant `view()` for Memory-Mapped Indexes

**Motivation:** ISCC indexing uses dozens of memory-mapped shards. Opening a 55-shard
index (18 GB, 100M vectors) took ~15.3 seconds because `view()` performs two O(n)
startup operations:

1. **`vectors_lookup_` population** -- Pre-computes a pointer for every vector in
    the memory-mapped file (8 bytes per vector, ~800 MB for 100M vectors).
1. **`reindex_keys_()` call** -- Builds a hash map for key-to-slot lookups by
    scanning the entire HNSW graph.

These are useful for general read-write workloads, but unnecessary for our read-only
memory-mapped shards where vectors are contiguous and offsets are trivially computable
as `base + stride * slot`.

**Changes (3 patches in `index_dense.hpp` and `lib.cpp`):**

- **Computed offsets for view mode:** Replaces the `vectors_lookup_` pointer table
    with stride-based arithmetic (`base + stride * slot`). Eliminates the O(n) loop
    and the per-vector pointer memory overhead entirely.

- **Lazy key reindexing:** Defers `slot_lookup_` population to first access via
    double-checked locking (`std::atomic<bool>` + mutex). The `view()` call returns
    immediately; key reindexing happens on the first `contains()`, `get()`, or
    `count()` call.

- **GIL release in Python bindings:** Wraps `load()` and `view()` with
    `py::gil_scoped_release` (when no progress callback is active), enabling
    thread-based parallelism for loading multiple shards concurrently.

**Benchmark results (stock usearch vs. ISCC fork):[^bench]**

| Configuration                              | Stock            | Fork    | Speedup |
| ------------------------------------------ | ---------------- | ------- | ------- |
| Single shard (315 MB, 1.8M vectors)        | 0.31 s           | 0.018 s | 17x     |
| 55 shards sequential (18 GB, 100M vectors) | 15.3 s           | 1.22 s  | 13x     |
| 55 shards parallel (8 threads)             | no speedup (GIL) | 0.49 s  | 31x     |

### Additional Patches

The fork also includes fixes for edge cases encountered in our workloads. These have
been reported upstream and may be resolved in future usearch releases.

**`Index.get()` for missing keys**
([#494](https://github.com/unum-cloud/usearch/issues/494)) -- `get()` on
non-existent keys returns uninitialized memory. Our patch adds a `contains()` check
and returns `None` for missing keys.

**`Index.search(count=0)` guard** -- `search()` with `count=0` triggers a
zero-width array allocation. Our patch adds validation at both Python and C++ levels.

**`serialized_length` Python binding**
([#683](https://github.com/unum-cloud/usearch/issues/683)) -- pybind11 cannot
resolve the default `serialization_config_t` argument. Our patch wraps it with a
lambda calling the no-argument overload.

**`IndexedKeys.__array__` NumPy 2.0 compatibility** -- Missing `copy` keyword
parameter causes `DeprecationWarning` in NumPy 2.0. Our patch adds and honors the
parameter.

## iscc-usearch Optimizations

### Bloom Filter for Key Rejection

`ShardedIndex` maintains a
[scalable bloom filter](https://en.wikipedia.org/wiki/Bloom_filter#Scalable_Bloom_filters)
(`ScalableBloomFilter` backed by `fastbloom-rs`) that provides O(1) rejection of
non-existent keys in `get()`, `contains()`, and `count()` operations.

Without the bloom filter, checking whether a key exists requires querying every shard
sequentially. With the bloom filter, keys that definitely don't exist are rejected
instantly without touching any shard.

**Design:**

- Chains multiple fixed-size bloom filters with progressively tighter false positive
    rates (geometric series with r=0.5) to support unlimited growth.
- Initial capacity: 10M elements, doubling per filter.
- Default false positive rate: 1%.
- Batch operations (`add_batch`, `contains_batch`) delegate to native Rust for
    throughput.
- Checks newest filter first (most likely location for recent keys).
- Persists to disk alongside shard files.

### Sharded Index Architecture

`ShardedIndex` splits vectors across multiple shard files to maintain consistent
write throughput and enable datasets larger than available RAM.

**Active shard** (one, fully loaded in RAM) handles writes. **View shards** (zero or
more, memory-mapped) handle reads. When the active shard exceeds the configured size
limit, it is saved to disk, reopened in view mode, and a fresh active shard is
created.

**Why this helps add throughput:** HNSW graph construction slows as the graph grows
because each insertion must search a larger neighborhood. A single usearch `Index`
averages ~11.7K vectors/sec over 1M inserts, with throughput declining as the graph
grows. Sharding resets the curve -- each shard stays small, maintaining ~13K
vectors/sec throughout.

| Metric                      | Single Index   | ShardedIndex (32 MB shards) |
| --------------------------- | -------------- | --------------------------- |
| Add throughput (1M vectors) | 11,682 vec/sec | 13,173 vec/sec              |
| Memory usage                | 324 MB         | 54 MB                       |

### Vectorized Search Merging

When searching across multiple shards, results from all shards are merged using
vectorized NumPy operations rather than Python loops:

- `np.concatenate` to stack results from all shards.
- `np.argsort(axis=1)` to sort each query row independently.
- Advanced indexing to gather sorted keys and distances without per-query iteration.
- Radius filtering applied as a vectorized mask across the entire batch.

Fast paths avoid allocation overhead in common cases (single source, no merge
needed).

### Shard Discovery Caching

Filesystem glob results (`shard_*.usearch`) are cached and only invalidated when
shards are created or rotated, eliminating repeated directory scans on every
operation.

### Lazy Key and Vector Iterators

`ShardedIndexedKeys` and `ShardedIndexedVectors` provide memory-efficient lazy
iteration across all shards. Keys and vectors are yielded shard-by-shard without
materializing everything in memory, enabling iteration over 100M+ vectors without
corresponding RAM usage.

### Numba JIT Compilation

The NPHD distance metric and vector padding/unpadding functions are compiled to
native code via Numba:

- **`@cfunc` metric:** The NPHD metric is compiled as a C-callable function pointer
    passed directly to usearch's C++ core, avoiding Python callback overhead on every
    distance computation.
- **`@njit(cache=True)` padding:** `pad_vectors` and `unpad_vectors` are compiled
    with result caching, so recompilation cost is paid only once.

### Fast-Path Key Lookups

`get()` and `contains()` follow a priority order to minimize shard iterations:

1. **Bloom filter** -- O(1) rejection if the key definitely doesn't exist.
1. **Active shard** -- Checked first since recent inserts are most likely here.
1. **View shards** -- Iterated with early termination once all requested keys are
    found.

## Performance Characteristics

### Query Latency vs. Shard Count

Query latency scales linearly with shard count since each shard is searched
independently and results are merged:

| Shards | Vectors | Avg query latency | QPS   |
| ------ | ------- | ----------------- | ----- |
| 1      | 916K    | 0.75 ms           | 1,343 |
| 10     | 9.2M    | 1.99 ms           | 504   |
| 25     | 22.9M   | 4.33 ms           | 231   |
| 50     | 45.8M   | 6.58 ms           | 152   |
| 100    | 91.6M   | 15.89 ms          | 63    |
| 109    | 100M    | 19.47 ms          | 51    |

*256-bit binary vectors, Hamming distance, M=16, efConstruction=32, ef=512, 128 MB shards.[^latency]*

### Cold Start

*55 shards (~315 MB each, 18 GB total, 100M vectors).*

| Configuration                            | Time   |
| ---------------------------------------- | ------ |
| 55-shard restore (usearch stock)         | 15.3 s |
| 55-shard restore (ISCC fork, sequential) | 1.22 s |
| 55-shard restore (ISCC fork, 8 threads)  | 0.49 s |

### Shard Size Tradeoffs

| Workload    | Recommended shard size | Rationale                              |
| ----------- | ---------------------- | -------------------------------------- |
| Write-heavy | 1/8 of available RAM   | More shards, consistent add throughput |
| Read-heavy  | 1/2 of available RAM   | Fewer shards, lower query latency      |
| Balanced    | 1/4 of available RAM   | Default recommendation                 |

[^bench]: Median of 5 runs after warmup. Stock usearch 2.23.0 (PyPI) vs. ISCC fork
    2.23.2. Windows 10, Intel i7-7700K, 64 GB RAM, Python 3.12. Benchmark script:
    `scripts/benchmark_view.py`.

[^latency]: Measured with a 109-shard dataset on the same hardware. Rows up to 55 shards
    reproduced independently.
