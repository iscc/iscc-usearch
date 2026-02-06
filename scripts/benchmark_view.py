"""Benchmark usearch view() performance on shard files.

Measures single-shard and multi-shard view() times. Useful for comparing stock
usearch against the patched ISCC fork.

Usage:
    uv run python scripts/benchmark_view.py <shard_directory>
    uv run python scripts/benchmark_view.py <shard_directory> --threads 4 --runs 10
"""

import argparse
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from usearch.index import Index

SHARD_PATTERN = "shard_*.usearch"


def get_shard_paths(directory):
    """Return sorted list of shard file paths in directory."""
    paths = sorted(directory.glob(SHARD_PATTERN))
    if not paths:
        print(f"Error: no files matching '{SHARD_PATTERN}' in {directory}", file=sys.stderr)
        sys.exit(1)
    return paths


def view_shard(path):
    """Open a single shard in view (memory-mapped) mode."""
    meta = Index.metadata(str(path))
    idx = Index(ndim=meta["dimensions"], metric=meta["kind_metric"], dtype=meta["kind_scalar"])
    idx.view(str(path))
    return idx


def bench(fn, warmup_runs, timed_runs):
    """Run fn with warmup and return list of elapsed times."""
    for _ in range(warmup_runs):
        result = fn()
        del result
        gc.collect()

    times = []
    for _ in range(timed_runs):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        del result

    return times


def median(values):
    """Return median of a list of values."""
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def fmt(seconds):
    """Format seconds for display."""
    if seconds < 0.1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.3f} s"


def main():
    """Run view() benchmarks on shard files."""
    parser = argparse.ArgumentParser(description="Benchmark usearch view() on shard files.")
    parser.add_argument("directory", type=Path, help="Directory containing shard_*.usearch files")
    parser.add_argument("--threads", type=int, default=8, help="Thread count for parallel test (default: 8)")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs (default: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs (default: 1)")
    args = parser.parse_args()

    import usearch

    print(f"usearch version: {usearch.__version__}")
    print(f"usearch location: {usearch.__file__}")
    print(f"Python: {sys.version}")
    print()

    shard_paths = get_shard_paths(args.directory)
    num_shards = len(shard_paths)
    first_shard = shard_paths[0]
    shard_size_mb = os.path.getsize(first_shard) / (1024 * 1024)

    meta = Index.metadata(str(first_shard))
    print(f"Shard directory: {args.directory}")
    print(f"Shards: {num_shards}, {shard_size_mb:.0f} MB each")
    print(f"Index config: ndim={meta['dimensions']}, metric={meta['kind_metric']}")
    print(f"Warmup: {args.warmup}, Timed runs: {args.runs}")
    print()

    # Single shard
    print(f"[1/3] Single shard view() ({shard_size_mb:.0f} MB) ...")
    times = bench(lambda: view_shard(first_shard), args.warmup, args.runs)
    print(f"  Median: {fmt(median(times))}  (runs: {', '.join(fmt(t) for t in times)})")
    print()

    # Sequential all shards
    print(f"[2/3] {num_shards}-shard sequential view() ...")
    times = bench(lambda: [view_shard(p) for p in shard_paths], args.warmup, args.runs)
    print(f"  Median: {fmt(median(times))}  (runs: {', '.join(fmt(t) for t in times)})")
    print()

    # Parallel all shards
    num_threads = args.threads
    print(f"[3/3] {num_shards}-shard parallel view() ({num_threads} threads) ...")

    def parallel_view():
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            return list(pool.map(view_shard, shard_paths))

    times = bench(parallel_view, args.warmup, args.runs)
    print(f"  Median: {fmt(median(times))}  (runs: {', '.join(fmt(t) for t in times)})")
    print()


if __name__ == "__main__":
    main()
