# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Batch `add()` with duplicate keys no longer corrupts index state. Duplicate keys (within the
    active shard) are silently skipped instead of raising `RuntimeError` with partial commit.
    Fixes inconsistent `size`, bloom filter, and dirty counter after failed batch adds
    ([#21](https://github.com/iscc/iscc-usearch/issues/21),
    [iscc/usearch#6](https://github.com/iscc/usearch/issues/6))

### Changed

- Index saves use buffer-then-write strategy for lower IOPS and power-loss durability.
    Serializes to an in-memory buffer, writes it to disk via `os.write()`, flushes with `fdatasync`,
    then atomically renames. Reduces write syscalls from ~3N to 3 (where N = number of vectors)
    and adds durability guarantees that were previously absent.
- Bump `usearch-iscc` to 2.24.4

## [0.6.1] - 2026-03-10

### Fixed

- `ShardedIndex.search()` raises `AxisError` when merging results from view and active shards
    with a single-row 2D query of shape `(1, ndim)` ([#22](https://github.com/iscc/iscc-usearch/issues/22))

## [0.6.0] - 2026-02-22

### Added

- `dirty` write counter property on `NphdIndex` and `ShardedIndex` for caller-driven flush policies
    ([#16](https://github.com/iscc/iscc-usearch/issues/16))
- "For Coding Agents" reference page in documentation with architecture map, decision dispatch tables,
    constraints catalog, and task recipes

### Fixed

- Accept `bytes` and `bytearray` in `pad_vectors()` per-element loop ([#14](https://github.com/iscc/iscc-usearch/issues/14))
- Only count existing keys in `dirty` counter on `remove()` ([#16](https://github.com/iscc/iscc-usearch/issues/16))
- Generate API reference markdown from source for `llms-full.txt` instead of raw mkdocstrings directives

### Changed

- Reduce logging verbosity: default `timer` level from INFO to DEBUG, reserve INFO for aggregate
    summaries like shard load/save counts ([#15](https://github.com/iscc/iscc-usearch/issues/15))
- Bump `usearch-iscc` minimum version to 2.24.2
- Move docs `overrides` directory into `docs/overrides` for cleaner project root
- Make `shard_size`, `connectivity`, `expansion_add`, `expansion_search` explicit parameters on
    `ShardedNphdIndex.__init__` (fixes griffe doc generation warnings)
- Resolve all `ty` type checker warnings caused by upstream type stubs and type narrowing gaps
- Match `NphdIndex.search()` signature to parent `Index.search()` (fixes LSP violation)

## [0.5.0] - 2026-02-20

### Changed

- Replace custom Numba NPHD metric with native usearch-iscc `MetricKind.NPHD` for 2-3x faster
    add and search operations
- Drop `numba` dependency (reduces install size and complexity)
- Rewrite `pad_vectors`/`unpad_vectors` as plain Python/NumPy (no JIT compilation)

## [0.4.0] - 2026-02-19

### Added

- `remove()` method for all sharded index variants — tombstone-based deletion for view shard entries,
    lazy USearch deletion for active shard entries
- `upsert()` method for all sharded index variants — insert-or-update with last-write-wins batch
    deduplication
- `compact()` method for all sharded index variants — rebuilds view shards excluding tombstoned and
    cross-shard duplicate entries to reclaim space
- `__delitem__` support (`del index[key]`) as alias for `remove()`
- Dedup-aware `keys` and `vectors` iterators — active shard authoritative, view shard entries filtered
    by tombstones and cross-shard overlap
- Cross-shard search result suppression — tombstoned and stale view shard copies are filtered from
    search results with adaptive oversampling
- Persistent tombstone tracking (`tombstones.npy`) — deletion state survives save/load cycles

### Fixed

- Widen distance tolerance for HNSW approximate search test on macOS

## [0.3.0] - 2026-02-17

### Added

- `read_only` mode for all sharded index variants — opens all shards as memory-mapped views, blocking
    write operations while allowing full read access
- `add_once()` method for skip-if-exists semantics on all sharded index variants
- `reset()` method for releasing resources without deleting files on disk

### Fixed

- `expansion_search` setter now propagates to view shards for consistent search quality
- Batch `search()` with mixed-length vectors in `ShardedNphdIndex`
- `save()` now raises `TypeError` if a path argument is passed to `ShardedIndex`

### Changed

- `ShardedIndex128.add()` accepts `list[bytes]` as batch keys

## [0.2.1] - 2026-02-16

### Changed

- Shard and bloom filter saves use atomic write (temp file + rename) to prevent corruption
    from interrupted writes

### Fixed

- Fix type error in `ShardedNphdIndex.get()` causing CI type check failure

## [0.2.0] - 2026-02-08

### Added

- `ShardedIndex128` — sharded index with 128-bit UUID keys (`bytes(16)` / `np.dtype('V16')`)
- `ShardedNphdIndex128` — sharded NPHD index with 128-bit UUID keys for variable-length vectors
- `_UuidKeyMixin` providing key-handling hooks, validation, and dispatch for all 128-bit subclasses
- `ScalableBloomFilter` support for `bytes` keys (`add`, `contains`, `add_batch`, `contains_batch`)
- Strict validation on 128-bit key operations — wrong key type, length, or dtype raises `ValueError`
- `upsert()` support for 128-bit UUID keys (single `bytes(16)` and batch `V16` ndarray)
- Python 3.10 and 3.11 support

### Changed

- Extract 6 key-handling hook methods on `ShardedIndex` (`_is_single_key`, `_bloom_key`,
    `_bloom_keys`, `_normalize_batch_keys`, `_shard_batch_keys`, `_key_dtype`) for subclass
    customization
- Remove UUID workaround hooks (`_iter_shard_vectors`, `_get_shard_vector`, `_register_view_shard`,
    `_search_view_shards`) — upstream usearch now supports `Index.vectors`, `Indexes.merge()`, and
    `Indexes.search()` for uuid-keyed indexes
- UUID sharded indexes now use C++-optimized `Indexes` multi-shard search instead of Python-side
    per-shard iteration
- Make `_merge_batch_matches` and `_apply_radius_filter` dtype-safe using `np.zeros_like` + mask-copy
    instead of `np.where(..., 0)` (V16 arrays do not support scalar 0 fill)
- Rewrite `ShardedIndexedKeys.__array__` to shard-aware concatenation preserving correct dtype
- Use `serialized_length` instead of `stats.allocated_bytes` for shard rotation threshold check
    (exactly matches on-disk file size and is faster)
- Amortize rotation size check to avoid O(n) `serialized_length` call on every `add()`

### Fixed

- Fix `size` property using wrong shard list — now sums `_viewed_indexes` which is always maintained
- Enable `serialized_length` property test (fixed upstream in usearch-iscc fork)

## [0.1.0] - 2026-02-06

Initial release of iscc-usearch.

### Added

- `NphdIndex` class extending USearch `Index` with variable-length binary vector support
- Normalized Prefix Hamming Distance (NPHD) custom metric compiled via Numba
- Length-prefixed padding for storing variable-length vectors in a fixed-dimension index
- `ShardedIndex` for scalable vector storage beyond available RAM
- `ShardedNphdIndex` combining NPHD metric with transparent sharding
- Idempotent `upsert` method for single and batch operations
- Bloom filter support for fast key membership checks
- Cross-shard `get`, `contains`, `count`, `keys`, and `vectors` operations
- Auto-detection of `ndim`/`max_dim` when opening existing indexes
- Timer context manager for logging operation durations
- Comprehensive test suite with 100% coverage requirement
- Documentation site using Zensical with Diataxis structure
- CI workflows for tests (Linux, macOS, Windows) and docs deployment
- Python 3.12, 3.13, and 3.14 support

[0.1.0]: https://github.com/iscc/iscc-usearch/releases/tag/0.1.0
[0.2.0]: https://github.com/iscc/iscc-usearch/compare/0.1.0...0.2.0
[0.2.1]: https://github.com/iscc/iscc-usearch/compare/0.2.0...0.2.1
[0.3.0]: https://github.com/iscc/iscc-usearch/compare/0.2.1...0.3.0
[0.4.0]: https://github.com/iscc/iscc-usearch/compare/0.3.0...0.4.0
[0.5.0]: https://github.com/iscc/iscc-usearch/compare/0.4.0...0.5.0
[0.6.0]: https://github.com/iscc/iscc-usearch/compare/0.5.0...0.6.0
