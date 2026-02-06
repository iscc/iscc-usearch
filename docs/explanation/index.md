# Explanation

Background material that explains *why* things work the way they do. Read these when you want to
understand the design decisions behind `iscc-usearch`.

- **[NPHD Metric](nphd-metric.md)** -- The math and properties of Normalized Prefix Hamming
    Distance.
- **[Architecture](architecture.md)** -- How variable-length vectors are bridged to fixed-dimension
    USearch through length-prefixed padding.
- **[Sharding Design](sharding-design.md)** -- Active vs. view shards, rotation, and bloom filter
    integration.
- **[Performance](performance.md)** -- Benchmarks, fork patches, and tuning guidance.
