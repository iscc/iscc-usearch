---
icon: lucide/gauge
---

# Performance

`iscc-usearch` depends on a [patched USearch fork](https://github.com/iscc/usearch) (v2.23.3)
tuned for ISCC-specific workloads. USearch is a fast ANNS library; the fork adds support for heavy
sharding, read-mostly access patterns, and large memory-mapped indexes that ISCC indexing requires.

## Patched USearch fork

Pre-built wheels are hosted at <https://iscc.github.io/usearch/> for Python 3.12+ on macOS, Linux
(x86_64, aarch64), and Windows (AMD64, ARM64).

### Instant `view()` for memory-mapped indexes

ISCC indexing uses dozens of memory-mapped shards. Opening a 55-shard index (18 GB, 100M vectors)
took ~15.3 seconds because `view()` performs two O(n) startup operations:

1. **`vectors_lookup_` population** -- Pre-computes a pointer for every vector in the memory-mapped
    file (8 bytes per vector, ~800 MB for 100M vectors).
1. **`reindex_keys_()` call** -- Builds a hash map for key-to-slot lookups by scanning the entire
    HNSW graph.

Both are useful for read-write workloads but unnecessary for read-only memory-mapped shards where
vectors are contiguous and offsets are computable as `base + stride * slot`.

Three patches in `index_dense.hpp` and `lib.cpp`:

- **Computed offsets for view mode:** Replaces the `vectors_lookup_` pointer table with
    stride-based arithmetic (`base + stride * slot`). This removes the O(n) loop and per-vector
    pointer memory overhead.
- **Lazy key reindexing:** Defers `slot_lookup_` population to first access via double-checked
    locking (`std::atomic<bool>` + mutex). `view()` returns immediately; key reindexing happens on
    the first `contains()`, `get()`, or `count()` call.
- **GIL release in Python bindings:** Wraps `load()` and `view()` with `py::gil_scoped_release`
    (when no progress callback is active), so multiple shards can load concurrently from threads.

### View benchmark results (stock USearch vs. ISCC fork)[^bench]

| Configuration                              | Stock            | Fork    | Speedup |
| ------------------------------------------ | ---------------- | ------- | ------- |
| Single shard (315 MB, 1.8M vectors)        | 0.31 s           | 0.018 s | 17x     |
| 55 shards sequential (18 GB, 100M vectors) | 15.3 s           | 1.22 s  | 13x     |
| 55 shards parallel (8 threads)             | no speedup (GIL) | 0.49 s  | 31x     |

### Additional patches

The fork also fixes edge cases found in ISCC workloads. These have been reported upstream and may be
resolved in future USearch releases.

- **`Index.get()` for missing keys**
    ([#494](https://github.com/unum-cloud/usearch/issues/494)) -- `get()` on non-existent keys
    returns uninitialized memory. The patch adds a `contains()` check and returns `None` for missing
    keys.
- **`Index.search(count=0)` guard** -- `search()` with `count=0` triggers a zero-width array
    allocation. The patch adds validation at both Python and C++ levels.
- **`serialized_length` Python binding**
    ([#683](https://github.com/unum-cloud/usearch/issues/683)) -- pybind11 cannot resolve the
    default `serialization_config_t` argument. The patch wraps it with a lambda calling the
    no-argument overload.
- **`IndexedKeys.__array__` NumPy 2.0 compatibility** -- Missing `copy` keyword parameter causes
    `DeprecationWarning` in NumPy 2.0. The patch adds and honors the parameter.

## iscc-usearch optimizations

### Bloom filter for key rejection

`ShardedIndex` has a scalable bloom filter for O(1) rejection of non-existent keys in `get()`,
`contains()`, and `count()`. Without it, checking whether a key exists requires querying every shard
sequentially.

Design:

- Chains multiple fixed-size bloom filters with progressively tighter false positive rates
    (geometric series with r=0.5) to support unlimited growth.
- Initial capacity: 10M elements, doubling per filter.
- Default false positive rate: 1%.
- Batch operations delegate to native Rust (`fastbloom-rs`) for throughput.
- Checks newest filter first (most likely location for recent keys).

#### Bloom filter vs. native key lookups[^bloom]

Even with the patched fork's efficient key lookups in view mode, the bloom filter is 3-50x faster
for non-existent key rejection. Without bloom, miss latency scales linearly with shard count because
every shard must be checked. With bloom, miss latency stays constant at ~1 us regardless of shard
count.

**Single key `contains()` -- non-existent keys (bloom's strength):**

| Shards | Bloom  | Native  | Speedup |
| ------ | ------ | ------- | ------- |
| 1      | 1.5 us | 3.9 us  | 2.6x    |
| 3      | 0.9 us | 4.1 us  | 4.6x    |
| 5      | 1.4 us | 9.6 us  | 6.9x    |
| 10     | 0.6 us | 24.5 us | 40.8x   |
| 20     | 1.0 us | 52.5 us | 52.5x   |

**Single key `contains()` -- existing keys (bloom overhead):**

| Shards | Bloom   | Native  | Overhead |
| ------ | ------- | ------- | -------- |
| 1      | 3.0 us  | 2.1 us  | 1.4x     |
| 3      | 4.6 us  | 2.6 us  | 1.8x     |
| 5      | 6.9 us  | 5.5 us  | 1.3x     |
| 10     | 12.7 us | 11.6 us | ~1x      |
| 20     | 21.2 us | 27.8 us | 0.8x     |

**Batch `get()` (100 keys) -- non-existent keys:**

| Shards | Bloom   | Native   | Speedup |
| ------ | ------- | -------- | ------- |
| 1      | 33.8 us | 23.5 us  | ~1x     |
| 5      | 86.0 us | 193.4 us | 2.3x    |
| 10     | 71.6 us | 462.5 us | 6.5x    |
| 20     | 65.9 us | 812.2 us | 12.3x   |

For existing keys, the bloom filter adds ~1.3-1.8x overhead at low shard counts but becomes neutral
or faster at higher counts where Python-level shard iteration dominates. The bloom filter is enabled
by default and recommended for all sharded indexes.

### Sharded index architecture

HNSW graph construction slows as the graph grows. A single USearch `Index` averages ~11.7K
vectors/sec over 1M inserts, with throughput dropping as the graph grows. Sharding resets the curve:
each shard stays small, holding ~13K vectors/sec throughout.

| Metric                   | Single Index   | ShardedIndex (32 MB shards) |
| ------------------------ | -------------- | --------------------------- |
| Add throughput (1M vecs) | 11,682 vec/sec | 13,173 vec/sec              |
| Memory usage             | 324 MB         | 54 MB                       |

### Vectorized search merging

Multi-shard search results are merged with vectorized NumPy operations:

- `np.concatenate` stacks results from all shards.
- `np.argsort(axis=1)` sorts each query row independently.
- Advanced indexing gathers sorted keys and distances without per-query iteration.
- Radius filtering is applied as a vectorized mask across the entire batch.

A fast path skips allocation when only one shard returns results.

### Shard discovery caching

Filesystem glob results (`shard_*.usearch`) are cached and only invalidated when shards are created
or rotated. This avoids repeated directory scans on every operation.

### Lazy key and vector iterators

`ShardedIndexedKeys` and `ShardedIndexedVectors` iterate lazily across all shards. Keys and vectors
are yielded shard-by-shard without loading everything into memory, so you can iterate over 100M+
vectors without proportional RAM usage.

### Numba JIT compilation

- **`@cfunc` metric:** The NPHD metric is compiled to a C-callable function pointer passed
    directly to USearch's C++ core, bypassing Python callback overhead on every distance computation.
- **`@njit(cache=True)` padding:** `pad_vectors` and `unpad_vectors` are compiled with result
    caching, so the compilation cost is paid only once.

### Fast-path key lookups

`get()` and `contains()` check shards in priority order to minimize iterations:

1. **Bloom filter** -- O(1) rejection if the key definitely doesn't exist.
1. **Active shard** -- Checked first since recent inserts are most likely here.
1. **View shards** -- Iterated with early termination once all requested keys are found.

## Performance characteristics

### Query latency vs. shard count

Query latency scales linearly with shard count because each shard is searched independently and
results are merged:

| Shards | Vectors | Avg query latency | QPS   |
| ------ | ------- | ----------------- | ----- |
| 1      | 916K    | 0.75 ms           | 1,343 |
| 10     | 9.2M    | 1.99 ms           | 504   |
| 25     | 22.9M   | 4.33 ms           | 231   |
| 50     | 45.8M   | 6.58 ms           | 152   |
| 100    | 91.6M   | 15.89 ms          | 63    |
| 109    | 100M    | 19.47 ms          | 51    |

*256-bit binary vectors, Hamming distance, M=16, efConstruction=32, ef=512, 128 MB shards.*[^latency]

### Cold start

*55 shards (~315 MB each, 18 GB total, 100M vectors).*

| Configuration                            | Time   |
| ---------------------------------------- | ------ |
| 55-shard restore (usearch stock)         | 15.3 s |
| 55-shard restore (ISCC fork, sequential) | 1.22 s |
| 55-shard restore (ISCC fork, 8 threads)  | 0.49 s |

## Tuning guidance

### Shard size

| Workload    | Recommended shard size | Rationale                              |
| ----------- | ---------------------- | -------------------------------------- |
| Write-heavy | 1/8 of available RAM   | More shards, consistent add throughput |
| Read-heavy  | 1/2 of available RAM   | Fewer shards, lower query latency      |
| Balanced    | 1/4 of available RAM   | Default recommendation                 |

### HNSW parameters

| Parameter          | Default | Effect                                              |
| ------------------ | ------- | --------------------------------------------------- |
| `connectivity`     | 16      | Higher = better recall, more memory, slower inserts |
| `expansion_add`    | 128     | Higher = better graph quality, slower inserts       |
| `expansion_search` | 64      | Higher = better recall, slower queries              |

The defaults work well for most workloads. Increase `expansion_search` if recall is too low;
increase `connectivity` for better recall at the cost of memory.

[^bench]: Median of 5 runs after warmup. Stock usearch 2.23.0 (PyPI) vs. ISCC fork 2.23.2
    (current fork is 2.23.3). Windows 10, Intel i7-7700K, 64 GB RAM, Python 3.12.

[^bloom]: Median of 2,000 iterations per operation. 5,000 vectors/shard, 256D float32. Same
    hardware as above. Reproducible via `uv run python scripts/benchmark_bloom_vs_native.py`.

[^latency]: Measured with a 109-shard dataset on the same hardware. Rows up to 55 shards
    reproduced independently.
