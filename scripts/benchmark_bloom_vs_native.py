"""Benchmark bloom filter vs native USearch key lookups for sharded indexes.

Compares performance of key lookup operations (contains, get, count) with
and without bloom filter across varying shard counts. Since the patched
usearch fork now supports efficient key lookups in view mode, this benchmark
quantifies whether the bloom filter still provides a meaningful advantage.

Usage:
    uv run python scripts/benchmark_bloom_vs_native.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from iscc_usearch import ShardedIndex

# Benchmark parameters
VECTORS_PER_SHARD = 5_000
NDIM = 256
ITERATIONS_SINGLE = 2_000
ITERATIONS_BATCH = 200
BATCH_SIZE = 100
SHARD_COUNTS = [1, 3, 5, 10, 20]
WARMUP_ITERATIONS = 100


def format_time(seconds: float) -> str:
    """Format time in human-readable units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def create_sharded_index(
    path: Path, num_shards: int, vectors_per_shard: int, bloom: bool
) -> tuple[ShardedIndex, list[int]]:
    """Create a sharded index with a controlled number of shards.

    Forces shard rotation by using a tiny shard size, then reloads with proper
    settings to get a realistic view-mode benchmark.
    """
    all_keys = []

    # Use a tiny shard_size to force rotations at the right points
    idx = ShardedIndex(
        ndim=NDIM,
        path=path,
        shard_size=1,  # Force rotation after first add
        bloom_filter=bloom,
    )

    for shard_idx in range(num_shards):
        vectors = np.random.rand(vectors_per_shard, NDIM).astype(np.float32)
        start_key = shard_idx * vectors_per_shard
        keys = list(range(start_key, start_key + vectors_per_shard))
        all_keys.extend(keys)
        idx.add(keys, vectors)

    idx.save()

    # Reload with a large shard_size so existing shards stay as view shards
    idx2 = ShardedIndex(
        ndim=NDIM,
        path=path,
        shard_size=1024 * 1024 * 1024,
        bloom_filter=bloom,
    )

    return idx2, all_keys


def benchmark_single_contains(idx: ShardedIndex, keys: list[int], iterations: int) -> float:
    """Benchmark single-key contains. Returns median latency per operation."""
    times = []
    n = len(keys)
    for i in range(iterations):
        key = keys[i % n]
        start = time.perf_counter()
        idx.contains(key)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def benchmark_batch_contains(idx: ShardedIndex, keys: list[int], batch_size: int, iterations: int) -> float:
    """Benchmark batch contains. Returns median latency per batch operation."""
    times = []
    n = len(keys)
    for i in range(iterations):
        start_i = (i * batch_size) % n
        batch = keys[start_i : start_i + batch_size]
        if len(batch) < batch_size:
            batch = batch + keys[: batch_size - len(batch)]
        start = time.perf_counter()
        idx.contains(batch)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def benchmark_single_get(idx: ShardedIndex, keys: list[int], iterations: int) -> float:
    """Benchmark single-key get. Returns median latency per operation."""
    times = []
    n = len(keys)
    for i in range(iterations):
        key = keys[i % n]
        start = time.perf_counter()
        idx.get(key)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def benchmark_batch_get(idx: ShardedIndex, keys: list[int], batch_size: int, iterations: int) -> float:
    """Benchmark batch get. Returns median latency per batch operation."""
    times = []
    n = len(keys)
    for i in range(iterations):
        start_i = (i * batch_size) % n
        batch = keys[start_i : start_i + batch_size]
        if len(batch) < batch_size:
            batch = batch + keys[: batch_size - len(batch)]
        start = time.perf_counter()
        idx.get(batch)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def benchmark_single_count(idx: ShardedIndex, keys: list[int], iterations: int) -> float:
    """Benchmark single-key count. Returns median latency per operation."""
    times = []
    n = len(keys)
    for i in range(iterations):
        key = keys[i % n]
        start = time.perf_counter()
        idx.count(key)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def run_scenario(
    label: str,
    shard_count: int,
    keys_existing: list[int],
    keys_missing: list[int],
    idx_bloom: ShardedIndex,
    idx_nobloom: ShardedIndex,
) -> dict:
    """Run all benchmarks for one shard count scenario."""
    results = {}

    # --- Warmup ---
    for key in keys_existing[:WARMUP_ITERATIONS]:
        idx_bloom.contains(key)
        idx_nobloom.contains(key)
    for key in keys_missing[:WARMUP_ITERATIONS]:
        idx_bloom.contains(key)
        idx_nobloom.contains(key)

    # --- Single contains: non-existent keys (bloom's best case) ---
    bloom_time = benchmark_single_contains(idx_bloom, keys_missing, ITERATIONS_SINGLE)
    native_time = benchmark_single_contains(idx_nobloom, keys_missing, ITERATIONS_SINGLE)
    results["contains_miss_single"] = (bloom_time, native_time)

    # --- Single contains: existing keys (bloom overhead) ---
    bloom_time = benchmark_single_contains(idx_bloom, keys_existing, ITERATIONS_SINGLE)
    native_time = benchmark_single_contains(idx_nobloom, keys_existing, ITERATIONS_SINGLE)
    results["contains_hit_single"] = (bloom_time, native_time)

    # --- Single get: non-existent keys ---
    bloom_time = benchmark_single_get(idx_bloom, keys_missing, ITERATIONS_SINGLE)
    native_time = benchmark_single_get(idx_nobloom, keys_missing, ITERATIONS_SINGLE)
    results["get_miss_single"] = (bloom_time, native_time)

    # --- Single get: existing keys ---
    bloom_time = benchmark_single_get(idx_bloom, keys_existing, ITERATIONS_SINGLE)
    native_time = benchmark_single_get(idx_nobloom, keys_existing, ITERATIONS_SINGLE)
    results["get_hit_single"] = (bloom_time, native_time)

    # --- Single count: non-existent keys ---
    bloom_time = benchmark_single_count(idx_bloom, keys_missing, ITERATIONS_SINGLE)
    native_time = benchmark_single_count(idx_nobloom, keys_missing, ITERATIONS_SINGLE)
    results["count_miss_single"] = (bloom_time, native_time)

    # --- Batch contains: non-existent keys ---
    bloom_time = benchmark_batch_contains(idx_bloom, keys_missing, BATCH_SIZE, ITERATIONS_BATCH)
    native_time = benchmark_batch_contains(idx_nobloom, keys_missing, BATCH_SIZE, ITERATIONS_BATCH)
    results["contains_miss_batch"] = (bloom_time, native_time)

    # --- Batch contains: existing keys ---
    bloom_time = benchmark_batch_contains(idx_bloom, keys_existing, BATCH_SIZE, ITERATIONS_BATCH)
    native_time = benchmark_batch_contains(idx_nobloom, keys_existing, BATCH_SIZE, ITERATIONS_BATCH)
    results["contains_hit_batch"] = (bloom_time, native_time)

    # --- Batch get: non-existent keys ---
    bloom_time = benchmark_batch_get(idx_bloom, keys_missing, BATCH_SIZE, ITERATIONS_BATCH)
    native_time = benchmark_batch_get(idx_nobloom, keys_missing, BATCH_SIZE, ITERATIONS_BATCH)
    results["get_miss_batch"] = (bloom_time, native_time)

    # --- Batch get: existing keys ---
    bloom_time = benchmark_batch_get(idx_bloom, keys_existing, BATCH_SIZE, ITERATIONS_BATCH)
    native_time = benchmark_batch_get(idx_nobloom, keys_existing, BATCH_SIZE, ITERATIONS_BATCH)
    results["get_hit_batch"] = (bloom_time, native_time)

    return results


def print_results_table(all_results: dict[int, dict]) -> None:
    """Print a summary comparison table."""
    operations = [
        ("contains_miss_single", "contains(miss) single"),
        ("contains_hit_single", "contains(hit)  single"),
        ("get_miss_single", "get(miss)      single"),
        ("get_hit_single", "get(hit)       single"),
        ("count_miss_single", "count(miss)    single"),
        ("contains_miss_batch", f"contains(miss) batch({BATCH_SIZE})"),
        ("contains_hit_batch", f"contains(hit)  batch({BATCH_SIZE})"),
        ("get_miss_batch", f"get(miss)      batch({BATCH_SIZE})"),
        ("get_hit_batch", f"get(hit)       batch({BATCH_SIZE})"),
    ]

    shard_counts = sorted(all_results.keys())

    for op_key, op_label in operations:
        print(f"\n  {op_label}")
        print(f"  {'Shards':>8} {'Bloom':>12} {'Native':>12} {'Speedup':>10} {'Winner':>10}")
        print(f"  {'─' * 8} {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 10}")

        for sc in shard_counts:
            bloom_t, native_t = all_results[sc][op_key]
            if native_t > 0 and bloom_t > 0:
                speedup = native_t / bloom_t
                winner = "bloom" if speedup > 1.05 else ("native" if speedup < 0.95 else "tie")
            else:
                speedup = float("nan")
                winner = "?"
            print(f"  {sc:>8} {format_time(bloom_t):>12} {format_time(native_t):>12} {speedup:>9.2f}x {winner:>10}")


def print_verdict(all_results: dict[int, dict]) -> None:
    """Print final verdict and recommendation."""
    print("\n" + "=" * 74)
    print("  VERDICT")
    print("=" * 74)

    # Collect all speedups for miss operations (bloom's strength)
    miss_speedups = {}
    for sc, results in all_results.items():
        speedups = []
        for key in ["contains_miss_single", "get_miss_single", "count_miss_single"]:
            bloom_t, native_t = results[key]
            if bloom_t > 0:
                speedups.append(native_t / bloom_t)
        miss_speedups[sc] = statistics.mean(speedups) if speedups else 1.0

    # Collect hit overheads (bloom's weakness)
    hit_overheads = {}
    for sc, results in all_results.items():
        overheads = []
        for key in ["contains_hit_single", "get_hit_single"]:
            bloom_t, native_t = results[key]
            if native_t > 0:
                overheads.append(bloom_t / native_t)
        hit_overheads[sc] = statistics.mean(overheads) if overheads else 1.0

    print("\n  Summary by shard count:")
    print(f"  {'Shards':>8} {'Miss Speedup':>14} {'Hit Overhead':>14} {'Net Benefit':>12}")
    print(f"  {'─' * 8} {'─' * 14} {'─' * 14} {'─' * 12}")
    for sc in sorted(miss_speedups.keys()):
        ms = miss_speedups[sc]
        ho = hit_overheads[sc]
        # Net benefit assuming 50/50 hit/miss ratio
        net = (ms + (1.0 / ho)) / 2.0
        label = "positive" if net > 1.1 else ("negative" if net < 0.9 else "neutral")
        print(f"  {sc:>8} {ms:>13.2f}x {ho:>13.2f}x {label:>12}")

    # Find threshold where bloom becomes worthwhile
    significant_shards = [sc for sc, ms in miss_speedups.items() if ms > 1.5]

    print("\n  Analysis:")
    print(
        f"  - Bloom filter provides >1.5x miss speedup at: "
        f"{significant_shards if significant_shards else 'no'} shard counts"
    )

    max_miss_speedup = max(miss_speedups.values())
    max_hit_overhead = max(hit_overheads.values())
    print(f"  - Peak miss speedup: {max_miss_speedup:.2f}x")
    print(f"  - Peak hit overhead: {max_hit_overhead:.2f}x")

    if max_miss_speedup < 2.0:
        print("\n  RECOMMENDATION: DROP the bloom filter")
        print("  Reason: With the patched usearch fork supporting efficient key lookups")
        print("  in view mode, native hash table lookups across shards are fast enough.")
        print("  The bloom filter adds complexity (bloom.py, persistence, sync logic)")
        print("  without providing a compelling performance advantage.")
    elif max_miss_speedup < 5.0 and max(miss_speedups.get(sc, 1.0) for sc in [1, 3, 5]) < 2.0:
        print("\n  RECOMMENDATION: DROP the bloom filter (marginal benefit)")
        print("  Reason: Bloom filter only helps at very high shard counts which are")
        print("  uncommon in practice. The complexity cost outweighs the marginal gain.")
    else:
        print("\n  RECOMMENDATION: KEEP the bloom filter")
        print("  Reason: Bloom filter provides significant speedup for miss lookups,")
        print("  especially at higher shard counts.")


def main():
    print("=" * 74)
    print("  Bloom Filter vs Native USearch Key Lookups")
    print("  (Post-fork: all shards support efficient key lookups)")
    print("=" * 74)
    print(f"\n  Config: {VECTORS_PER_SHARD:,} vectors/shard, {NDIM}D float32")
    print(f"  Single iterations: {ITERATIONS_SINGLE:,}")
    print(f"  Batch iterations:  {ITERATIONS_BATCH:,}, batch size: {BATCH_SIZE}")

    all_results = {}

    for num_shards in SHARD_COUNTS:
        total_vectors = num_shards * VECTORS_PER_SHARD
        print(f"\n{'─' * 74}")
        print(f"  Setting up {num_shards} shards ({total_vectors:,} total vectors)...")

        # Use ignore_cleanup_errors for Windows (memory-mapped files hold locks)
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            bloom_path = Path(tmpdir) / "bloom"
            nobloom_path = Path(tmpdir) / "nobloom"

            # Create indexes
            print("  Creating bloom index...")
            idx_bloom, all_keys = create_sharded_index(bloom_path, num_shards, VECTORS_PER_SHARD, bloom=True)
            print(f"    Shards: {idx_bloom.shard_count}, Vectors: {len(idx_bloom):,}")

            print("  Creating native index...")
            idx_nobloom, _ = create_sharded_index(nobloom_path, num_shards, VECTORS_PER_SHARD, bloom=False)
            print(f"    Shards: {idx_nobloom.shard_count}, Vectors: {len(idx_nobloom):,}")

            # Prepare test keys
            existing_keys = all_keys.copy()
            np.random.shuffle(existing_keys)
            max_key = max(all_keys)
            missing_keys = list(range(max_key + 1_000_000, max_key + 1_100_000))

            # Run benchmarks
            print("  Running benchmarks...")
            results = run_scenario(
                f"{num_shards} shards",
                num_shards,
                existing_keys,
                missing_keys,
                idx_bloom,
                idx_nobloom,
            )
            all_results[num_shards] = results

            # Print quick summary for this shard count
            bloom_miss, native_miss = results["contains_miss_single"]
            bloom_hit, native_hit = results["contains_hit_single"]
            miss_speedup = native_miss / bloom_miss if bloom_miss > 0 else 0
            hit_overhead = bloom_hit / native_hit if native_hit > 0 else 0
            print(f"  Result: miss speedup={miss_speedup:.2f}x, hit overhead={hit_overhead:.2f}x")

            # Release index references to help Windows cleanup
            del idx_bloom, idx_nobloom

    # Print full results
    print("\n" + "=" * 74)
    print("  DETAILED RESULTS")
    print("=" * 74)
    print_results_table(all_results)
    print_verdict(all_results)
    print()


if __name__ == "__main__":
    main()
