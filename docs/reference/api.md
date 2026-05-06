---
icon: lucide/book-open
description: API reference for NphdIndex, ShardedNphdIndex, ShardedIndex, ShardedIndex128, ShardedNphdIndex128, ScalableBloomFilter, and timer utility.
---

# API Reference

Auto-generated documentation for all public classes in `iscc-usearch`.

## NphdIndex

Single-file index for variable-length binary bit-vectors with NPHD metric.

::: iscc_usearch.NphdIndex
    options:
        show_source: false
        heading_level: 3
        members_order: source

## ShardedNphdIndex

Multi-shard index combining automatic sharding with NPHD support for variable-length vectors.

::: iscc_usearch.ShardedNphdIndex
    options:
        show_source: false
        heading_level: 3
        members_order: source

## ShardedIndex

Generic sharded index for any metric. Use `ShardedNphdIndex` for NPHD workloads.

::: iscc_usearch.ShardedIndex
    options:
        show_source: false
        heading_level: 3
        members_order: source

## ShardedIndex128

Sharded index with 128-bit UUID keys. Uses `bytes(16)` for single keys and `np.dtype('V16')`
arrays for batches.

::: iscc_usearch.ShardedIndex128
    options:
        show_source: false
        heading_level: 3
        members_order: source

## ShardedNphdIndex128

Sharded NPHD index with 128-bit UUID keys for variable-length vectors.

::: iscc_usearch.ShardedNphdIndex128
    options:
        show_source: false
        heading_level: 3
        members_order: source

## ScalableBloomFilter

Scalable bloom filter for efficient probabilistic key existence checks.

::: iscc_usearch.ScalableBloomFilter
    options:
        show_source: false
        heading_level: 3
        members_order: source

## timer

Context manager for timing operations with loguru integration.

::: iscc_usearch.timer
    options:
        show_source: false
        heading_level: 3
        members_order: source
