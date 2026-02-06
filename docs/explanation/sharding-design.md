---
icon: lucide/network
---

# Sharding Design

## The problem

HNSW graph construction slows as the graph grows because each insertion must search a larger
neighborhood. A single USearch `Index` averages ~11.7K vectors/sec over 1M inserts, with throughput
declining throughout. For datasets with hundreds of millions of vectors, this becomes a bottleneck.

Additionally, a single index file must fit in RAM for writes. Memory-mapping helps for reads, but
write-heavy workloads need the full graph loaded.

## Active shard vs. view shards

`ShardedIndex` splits storage into two tiers:

- **Active shard** (one, fully loaded in RAM): Handles all writes. Stays small for consistent
    insert throughput.
- **View shards** (zero or more, memory-mapped): Handle reads. Low memory footprint since the OS
    pages data in on demand.

```mermaid
stateDiagram-v2
    [*] --> Created: new shard
    Created --> Filling: add vectors
    Filling --> Full: size > shard_size
    Full --> Saved: save to disk
    Saved --> Viewed: reopen as mmap
    Viewed --> [*]

    note right of Filling: Active shard (RAM, read-write)
    note right of Viewed: View shard (mmap, read-only)
```

When the active shard exceeds the configured `shard_size`, it is:

1. Saved to disk as `shard_NNN.usearch`.
1. Reopened in view mode (memory-mapped, read-only).
1. Replaced by a fresh, empty active shard.

This rotation resets the HNSW insert curve, maintaining consistent throughput.

## Bloom filter integration

`ShardedIndex` maintains a `ScalableBloomFilter` that tracks all keys across all shards. This
enables O(1) rejection of non-existent keys in `get()`, `contains()`, and `count()`:

```mermaid
graph TD
    Q["get(key)"] --> BF{"Bloom filter<br/>contains(key)?"}
    BF -->|"Definitely no"| NONE["Return None"]
    BF -->|"Maybe yes"| AS{"Active shard<br/>contains(key)?"}
    AS -->|"Yes"| RA["Return from active"]
    AS -->|"No"| VS{"View shards<br/>(iterate)"}
    VS -->|"Found"| RV["Return from view shard"]
    VS -->|"Not found"| NONE2["Return None"]

    style BF fill:#fff3e0,stroke:#f57c00
    style NONE fill:#ffcdd2,stroke:#d32f2f
    style NONE2 fill:#ffcdd2,stroke:#d32f2f
    style RA fill:#c8e6c9,stroke:#388e3c
    style RV fill:#c8e6c9,stroke:#388e3c
```

Without the bloom filter, checking key existence requires querying every shard sequentially. With
the bloom filter, keys that definitely don't exist are rejected instantly.

The bloom filter is:

- Persisted alongside shard files as `bloom.isbf`.
- Kept in sync automatically as vectors are added.
- Rebuilt on demand via `rebuild_bloom()` if corrupted or missing.

## Search fan-out

Search queries are executed against all shards in parallel (via USearch's `Indexes` class for view
shards), then results are merged:

1. Query all view shards (via `Indexes.search()`).
1. Query the active shard.
1. Merge results using vectorized NumPy operations (concatenate, argsort, advanced indexing).
1. Return top-k results sorted by distance.

## Trade-offs

| Factor         | Fewer shards (large shard_size)    | More shards (small shard_size)     |
| -------------- | ---------------------------------- | ---------------------------------- |
| Query latency  | Lower (fewer shards to search)     | Higher (more shards to merge)      |
| Add throughput | Degrades as shard fills            | Stays consistent (frequent resets) |
| Memory usage   | Higher (large active shard in RAM) | Lower (small active shard)         |
| Disk I/O       | Less frequent rotation             | More frequent rotation             |

The default `shard_size` of 1 GB provides a balance for most workloads. Tune based on your
read/write ratio -- see [Performance](performance.md) for benchmark data.

## Append-only design

`ShardedIndex` is append-only by design:

- No `remove()`: View shards are read-only and USearch doesn't support efficient single-key
    deletion from memory-mapped files.
- No `clear()` / `reset()`: Would require coordinating across multiple shard files.
- No `copy()`: Would require deep-copying multiple memory-mapped files.

This keeps the implementation simple and predictable. For workloads that need updates, use
`NphdIndex` with `upsert()` (single-file only).
