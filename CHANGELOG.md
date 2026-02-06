# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
