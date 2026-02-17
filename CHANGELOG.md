# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `expansion_search` setter now propagates to view shards for consistent search quality
- Batch `search()` with mixed-length vectors in `ShardedNphdIndex`

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
