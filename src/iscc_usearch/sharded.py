"""
Sharded index implementation for scalable vector storage.

Provides a drop-in replacement for usearch Index with automatic sharding support.
Wraps multiple Index files (shards) for append-only storage that scales beyond
single-file limitations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from usearch.index import (
    BatchMatches,
    Index,
    Indexes,
    Matches,
    MetricKind,
    ScalarKind,
)

from iscc_usearch.utils import timer

__all__ = ["ShardedIndex"]

# Default shard size: 1GB
DEFAULT_SHARD_SIZE = 1024 * 1024 * 1024


class ShardedIndex:
    """Sharded vector index for scalable append-only storage.

    Wraps usearch Index/Indexes to provide automatic sharding when the active
    shard exceeds the configured size limit. Finished shards are memory-mapped
    (view mode) for efficient read-only access, while the active shard is
    fully loaded (load mode) for read-write operations.

    CONCURRENCY: Single-process only. No file locking. Use async/await within
    a single process for concurrent access.

    :param ndim: Number of vector dimensions
    :param metric: Distance metric (MetricKind or CompiledMetric)
    :param dtype: Scalar type for vectors (ScalarKind)
    :param connectivity: HNSW connectivity parameter (M)
    :param expansion_add: Search depth on insertions (efConstruction)
    :param expansion_search: Search depth on queries (ef)
    :param multi: Allow multiple vectors per key
    :param path: Directory path for shard storage (required)
    :param shard_size: Size limit in bytes before rotating shards (default 1GB)
    :param view: Load existing shards in view mode only (read-only)
    :param enable_key_lookups: Enable key-based lookups (get, contains)
    """

    def __init__(
        self,
        *,
        ndim: int,
        metric: MetricKind | Any = MetricKind.Cos,
        dtype: ScalarKind | str | None = None,
        connectivity: int | None = None,
        expansion_add: int | None = None,
        expansion_search: int | None = None,
        multi: bool = False,
        path: str | os.PathLike,
        shard_size: int = DEFAULT_SHARD_SIZE,
        view: bool = False,
        enable_key_lookups: bool = True,
    ) -> None:
        """Initialize a sharded index."""
        self._path = Path(path)
        self._shard_size = shard_size
        self._view_mode = view

        # Store config for creating new shards
        self._config: dict[str, Any] = {
            "ndim": ndim,
            "metric": metric,
            "dtype": dtype,
            "connectivity": connectivity,
            "expansion_add": expansion_add,
            "expansion_search": expansion_search,
            "multi": multi,
            "enable_key_lookups": enable_key_lookups,
        }

        # Initialize shard containers
        self._view_shards: Indexes | None = None
        self._active_shard: Index | None = None
        # Path to the active shard file (None if new/unsaved)
        self._active_shard_path: Path | None = None
        # Keep references to viewed Index objects to prevent garbage collection
        # (Indexes.merge only stores references, not copies)
        self._viewed_indexes: list[Index] = []

        # Create directory if needed
        self._path.mkdir(parents=True, exist_ok=True)

        # Discover existing shards
        existing_shards = self._discover_shards()

        if existing_shards:
            if view:
                self.view()
            else:
                self.load()
        elif not view:
            # Create first active shard
            self._active_shard = self._create_shard()

    # === Core Operations ===

    def add(
        self,
        keys: int | None | Any,
        vectors: NDArray[Any],
        *,
        copy: bool = True,
        threads: int = 0,
        log: str | bool = False,
        progress: Callable[[int, int], bool] | None = None,
    ) -> int | NDArray[np.uint64]:
        """Add vectors to the active shard, rotating if size exceeded.

        :param keys: Integer key(s) or None for auto-generation
        :param vectors: Vector or batch of vectors to add
        :param copy: Whether to copy vectors into index
        :param threads: Number of threads (0 = auto)
        :param log: Enable progress logging
        :param progress: Progress callback
        :return: Key(s) for added vectors
        :raises RuntimeError: If index is in view mode
        """
        if self._view_mode:
            raise RuntimeError("Cannot add to index opened in view mode")

        if self._active_shard is None:
            self._active_shard = self._create_shard()

        # Delegate to active shard
        result = self._active_shard.add(keys, vectors, copy=copy, threads=threads, log=log, progress=progress)

        # Check if rotation needed (after add completes)
        # Use stats.allocated_bytes as it better reflects actual disk size
        # than memory_usage (which includes pre-allocated capacity)
        if self._active_shard.stats.allocated_bytes > self._shard_size:
            self._rotate_shard()

        return result

    def search(
        self,
        vectors: NDArray[Any],
        count: int = 10,
        *,
        radius: float = float("inf"),
        threads: int = 0,
        exact: bool = False,
        log: str | bool = False,
        progress: Callable[[int, int], bool] | None = None,
    ) -> Matches | BatchMatches:
        """Search across all shards, merging and sorting results.

        :param vectors: Query vector or batch of vectors
        :param count: Maximum number of results per query
        :param radius: Maximum distance for results
        :param threads: Number of threads (0 = auto)
        :param exact: Perform exact search
        :param log: Enable progress logging
        :param progress: Progress callback
        :return: Matches for single query, BatchMatches for batch
        :raises ValueError: If count < 1
        """
        if count < 1:
            raise ValueError("count must be >= 1")

        vectors = np.asarray(vectors)
        is_single = vectors.ndim == 1

        results: list[Matches | BatchMatches] = []
        has_view_results = False

        # Search view shards (via Indexes - doesn't support radius parameter)
        if self._view_shards is not None and len(self._view_shards) > 0:
            view_results = self._view_shards.search(
                vectors, count=count, threads=threads, exact=exact, progress=progress
            )
            results.append(view_results)
            has_view_results = True

        # Search active shard (supports radius parameter)
        if self._active_shard is not None and len(self._active_shard) > 0:
            active_results = self._active_shard.search(
                vectors,
                count=count,
                radius=radius,
                threads=threads,
                exact=exact,
                log=log,
                progress=progress,
            )
            results.append(active_results)

        # Handle empty index
        if not results:
            return self._empty_results(vectors, count, is_single)

        # Single source - apply radius filter only if result is from view shards
        # (Indexes.search doesn't support radius, Index.search already applied it)
        if len(results) == 1:
            result = results[0]
            if radius < float("inf") and has_view_results:
                result = self._apply_radius_filter(result, radius, is_single)
            return result

        # Merge results from multiple sources (applies radius filter)
        return self._merge_search_results(results, count, radius, is_single)

    def get(
        self,
        keys: int | Any,
        dtype: Any = None,
    ) -> NDArray[Any] | list | None:
        """Retrieve vectors by key from active shard only.

        Note: View shards do not support get(). Only keys in the active shard
        can be retrieved.

        :param keys: Integer key(s) to lookup
        :param dtype: Optional data type for returned vectors
        :return: Vector(s) or None for missing keys
        """
        if self._active_shard is None:
            # Handle single vs multiple keys
            if isinstance(keys, int):
                return None
            return [None] * len(keys)

        return self._active_shard.get(keys, dtype=dtype)

    def contains(self, keys: int | Any) -> bool | NDArray[np.bool_]:
        """Check if keys exist in active shard only.

        Note: View shards do not support contains(). Only keys in the active
        shard are checked.

        :param keys: Integer key(s) to check
        :return: Boolean or array of booleans
        """
        if self._active_shard is None:
            if isinstance(keys, int):
                return False
            return np.zeros(len(keys), dtype=bool)

        return self._active_shard.contains(keys)

    def __contains__(self, keys: int | Any) -> bool | NDArray[np.bool_]:
        """Support 'in' operator."""
        return self.contains(keys)

    def count(self, keys: int | Any) -> int | NDArray[np.uint64]:
        """Count occurrences of keys in active shard only.

        Note: View shards do not support count(). Only keys in the active
        shard are counted.

        :param keys: Integer key(s) to count
        :return: Count or array of counts
        """
        if self._active_shard is None:
            if isinstance(keys, int):
                return 0
            return np.zeros(len(keys), dtype=np.uint64)

        return self._active_shard.count(keys)

    # === Persistence ===

    def save(
        self,
        path_or_buffer: str | os.PathLike | None = None,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None:
        """Save active shard to disk.

        :param path_or_buffer: Ignored (uses internal path management)
        :param progress: Progress callback
        """
        if self._active_shard is None or len(self._active_shard) == 0:
            return

        shard_path = self._get_active_shard_path()
        with timer(f"ShardedIndex save {shard_path.name}"):
            self._active_shard.save(shard_path, progress=progress)
        self._active_shard_path = shard_path

    def load(
        self,
        path_or_buffer: str | os.PathLike | None = None,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None:
        """Load shards from directory (active shard in load mode).

        :param path_or_buffer: Ignored (uses internal path management)
        :param progress: Progress callback
        """
        self._view_mode = False
        shard_files = self._discover_shards()

        if not shard_files:
            self._active_shard = self._create_shard()
            self._view_shards = None
            return

        with timer(f"ShardedIndex load {len(shard_files)} shards from {self._path}", log_start=True):
            # All but the last shard go into view mode
            view_paths = shard_files[:-1]
            active_path = shard_files[-1]

            # Create Indexes for view shards using workaround for usearch bug #643
            # (Indexes(paths=[...]) segfaults, so we restore individually and merge)
            # Keep references to prevent GC (Indexes.merge stores references, not copies)
            self._viewed_indexes = []
            if view_paths:
                view_shards = Indexes()
                for p in view_paths:
                    viewed = self._restore_shard(p, view=True)
                    assert viewed is not None
                    self._viewed_indexes.append(viewed)
                    view_shards.merge(viewed)
                self._view_shards = view_shards
            else:
                self._view_shards = None

            # Load active shard (writable) and track its path for save()
            active_shard = self._restore_shard(active_path, view=False)
            assert active_shard is not None
            self._active_shard = active_shard
            self._active_shard_path = active_path

        # Update config from loaded shard to ensure new shards match existing ones
        self._config["ndim"] = active_shard.ndim
        self._config["dtype"] = active_shard.dtype
        self._config["metric"] = active_shard.metric
        self._config["connectivity"] = active_shard.connectivity
        self._config["expansion_add"] = active_shard.expansion_add
        self._config["expansion_search"] = active_shard.expansion_search
        self._config["multi"] = active_shard.multi

    def view(
        self,
        path_or_buffer: str | os.PathLike | None = None,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None:
        """View shards from directory (all shards in view mode, read-only).

        :param path_or_buffer: Ignored (uses internal path management)
        :param progress: Progress callback
        """
        self._view_mode = True
        shard_files = self._discover_shards()

        if not shard_files:
            self._view_shards = None
            self._active_shard = None
            return

        with timer(f"ShardedIndex view {len(shard_files)} shards from {self._path}", log_start=True):
            # All shards in view mode (read-only)
            # Using workaround for usearch bug #643: Indexes(paths=[...]) segfaults
            # Keep references to prevent GC (Indexes.merge stores references, not copies)
            self._viewed_indexes = []
            view_shards = Indexes()
            for p in shard_files:
                viewed = self._restore_shard(p, view=True)
                assert viewed is not None
                self._viewed_indexes.append(viewed)
                view_shards.merge(viewed)
            self._view_shards = view_shards
            self._active_shard = None

    @staticmethod
    def restore(
        path: str | os.PathLike,
        view: bool = False,
        **kwargs: Any,
    ) -> ShardedIndex | None:
        """Restore a ShardedIndex from a directory.

        :param path: Directory containing shard files
        :param view: Open in view mode (read-only)
        :param kwargs: Additional arguments passed to constructor
        :return: Restored ShardedIndex or None if invalid
        """
        path = Path(path)
        if not path.exists() or not path.is_dir():
            return None

        # Discover shard files
        shard_files = sorted(
            path.glob("shard_*.usearch"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not shard_files:
            return None

        # Read metadata from first shard to get configuration
        meta = Index.metadata(str(shard_files[0]))
        if not meta:  # pragma: no cover (Index.metadata raises on invalid files)
            return None

        return ShardedIndex(
            ndim=meta["dimensions"],
            dtype=meta["kind_scalar"],
            metric=meta["kind_metric"],
            path=path,
            view=view,
            **kwargs,
        )

    @staticmethod
    def metadata(path: str | os.PathLike) -> dict | None:
        """Extract metadata from a sharded index directory.

        :param path: Directory containing shard files
        :return: Metadata dict or None if invalid
        """
        path = Path(path)
        if not path.exists() or not path.is_dir():
            return None

        shard_files = sorted(
            path.glob("shard_*.usearch"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not shard_files:
            return None

        return Index.metadata(str(shard_files[0]))

    # === Properties ===

    @property
    def size(self) -> int:
        """Total number of vectors across all shards."""
        total = 0
        if self._view_shards is not None:
            total += len(self._view_shards)
        if self._active_shard is not None:
            total += len(self._active_shard)
        return total

    def __len__(self) -> int:
        """Total number of vectors across all shards."""
        return self.size

    @property
    def ndim(self) -> int:
        """Vector dimensionality."""
        if self._active_shard is not None:
            return self._active_shard.ndim
        return self._config["ndim"]

    @property
    def dtype(self) -> ScalarKind:
        """Scalar type for vectors."""
        if self._active_shard is not None:
            return self._active_shard.dtype
        return self._config.get("dtype")

    @property
    def metric(self) -> MetricKind | Any:
        """Distance metric."""
        if self._active_shard is not None:
            return self._active_shard.metric
        return self._config.get("metric", MetricKind.Cos)

    @property
    def metric_kind(self) -> MetricKind:
        """Distance metric kind."""
        if self._active_shard is not None:
            return self._active_shard.metric_kind
        metric = self._config.get("metric", MetricKind.Cos)
        if isinstance(metric, MetricKind):
            return metric
        return metric.kind

    @property
    def connectivity(self) -> int:
        """HNSW connectivity parameter."""
        if self._active_shard is not None:
            return self._active_shard.connectivity
        return self._config.get("connectivity") or 16

    @property
    def expansion_add(self) -> int:
        """Expansion parameter for additions."""
        if self._active_shard is not None:
            return self._active_shard.expansion_add
        return self._config.get("expansion_add") or 128

    @expansion_add.setter
    def expansion_add(self, value: int) -> None:
        """Set expansion parameter for additions (active shard only)."""
        if self._active_shard is not None:
            self._active_shard.expansion_add = value
        self._config["expansion_add"] = value

    @property
    def expansion_search(self) -> int:
        """Expansion parameter for searches."""
        if self._active_shard is not None:
            return self._active_shard.expansion_search
        return self._config.get("expansion_search") or 64

    @expansion_search.setter
    def expansion_search(self, value: int) -> None:
        """Set expansion parameter for searches (active shard only)."""
        if self._active_shard is not None:
            self._active_shard.expansion_search = value
        self._config["expansion_search"] = value

    @property
    def multi(self) -> bool:
        """Whether multiple vectors per key are allowed."""
        if self._active_shard is not None:
            return self._active_shard.multi
        return self._config.get("multi", False)

    @property
    def path(self) -> Path:
        """Directory path for shard storage."""
        return self._path

    @property
    def shard_count(self) -> int:
        """Number of shard files."""
        return len(self._discover_shards())

    @property
    def memory_usage(self) -> int:
        """Estimated memory usage across all shards."""
        total = 0
        if self._active_shard is not None:
            total += self._active_shard.memory_usage
        for idx in self._viewed_indexes:
            total += idx.memory_usage
        return total

    @property
    def serialized_length(self) -> int:
        """Serialized length of active shard.

        Note: usearch pybind11 binding bug - property requires config arg that isn't exposed.
        Falls back to memory_usage on TypeError.
        """
        if self._active_shard is not None:
            try:
                return self._active_shard.serialized_length
            except TypeError:
                # usearch pybind11 bug: serialized_length missing default config arg
                return self._active_shard.memory_usage
        return 0

    @property
    def capacity(self) -> int:
        """Capacity of active shard."""
        if self._active_shard is not None:
            return self._active_shard.capacity
        return 0

    # === Not Supported ===

    def remove(self, *args: Any, **kwargs: Any) -> None:
        """Not supported - append-only design."""
        raise NotImplementedError("ShardedIndex is append-only; remove() not supported")

    def __delitem__(self, keys: Any) -> None:
        """Not supported - append-only design."""
        raise NotImplementedError("ShardedIndex is append-only; remove() not supported")

    def rename(self, *args: Any, **kwargs: Any) -> None:
        """Not supported - append-only design."""
        raise NotImplementedError("ShardedIndex is append-only; rename() not supported")

    def join(self, *args: Any, **kwargs: Any) -> None:
        """Not supported for sharded indexes."""
        raise NotImplementedError("join() not supported for ShardedIndex")

    def cluster(self, *args: Any, **kwargs: Any) -> None:
        """Not supported for sharded indexes."""
        raise NotImplementedError("cluster() not supported for ShardedIndex")

    def pairwise_distance(self, *args: Any, **kwargs: Any) -> None:
        """Not supported for sharded indexes."""
        raise NotImplementedError("pairwise_distance() not supported for ShardedIndex")

    def copy(self) -> None:
        """Not supported - too complex with multiple shards."""
        raise NotImplementedError("copy() not supported for ShardedIndex")

    def clear(self) -> None:
        """Not supported - would need to handle multiple files."""
        raise NotImplementedError("clear() not supported for ShardedIndex")

    def reset(self) -> None:
        """Not supported - would need to handle multiple files."""
        raise NotImplementedError("reset() not supported for ShardedIndex")

    @property
    def keys(self) -> None:
        """Not supported - would need to aggregate across shards."""
        raise NotImplementedError("keys property not supported for ShardedIndex")

    @property
    def vectors(self) -> None:
        """Not supported - would need to aggregate across shards."""
        raise NotImplementedError("vectors property not supported for ShardedIndex")

    # === Helper Methods ===

    def _create_shard(self) -> Index:
        """Create a new shard. Override in subclasses for custom shard types."""
        return Index(**self._config)

    def _restore_shard(self, path: Path, view: bool) -> Index | None:
        """Restore a shard from disk. Override in subclasses for custom shard types.

        When view=True, disables key lookups to skip expensive hash map population
        (~2x speedup). Safe because view shards only support search(), not get/contains.
        """
        meta = Index.metadata(str(path))
        if meta is None:  # pragma: no cover - shard files are always valid in practice
            return None
        idx = Index(
            ndim=meta["dimensions"],
            metric=meta["kind_metric"],
            dtype=meta["kind_scalar"],
            enable_key_lookups=not view,  # Disable for view shards (2x speedup)
        )
        if view:
            idx.view(str(path))
        else:
            idx.load(str(path))
        return idx

    def _discover_shards(self) -> list[Path]:
        """Return sorted list of shard_*.usearch files."""
        shards = list(self._path.glob("shard_*.usearch"))
        return sorted(shards, key=lambda p: int(p.stem.split("_")[1]))

    def _get_next_shard_number(self) -> int:
        """Get the next available shard number."""
        existing = self._discover_shards()
        if not existing:
            return 0
        last_num = int(existing[-1].stem.split("_")[1])
        return last_num + 1

    def _get_shard_path(self, shard_num: int) -> Path:
        """Generate path for a shard file."""
        return self._path / f"shard_{shard_num:03d}.usearch"

    def _get_active_shard_path(self) -> Path:
        """Get path for saving the active shard.

        Returns the tracked path if loaded from disk, otherwise the next available number.
        """
        if self._active_shard_path is not None:
            return self._active_shard_path
        existing = self._discover_shards()
        return self._get_shard_path(len(existing))

    def _rotate_shard(self) -> None:
        """Save current shard and create new one."""
        if self._active_shard is None:
            return

        # Use tracked path if loaded from disk (overwrites in-place),
        # otherwise use next available number for new shards
        if self._active_shard_path is not None:
            shard_path = self._active_shard_path
        else:
            existing_shards = self._discover_shards()
            shard_path = self._get_shard_path(len(existing_shards))

        # Save current active shard
        with timer(f"ShardedIndex rotate save {shard_path.name}"):
            self._active_shard.save(str(shard_path))
        # Clear tracked path since we're creating a new unsaved shard
        self._active_shard_path = None

        # Load the saved shard in view mode and merge into Indexes
        # Workaround for usearch bug #643: Indexes(paths=[...]) segfaults
        # Keep reference to prevent GC (Indexes.merge stores references, not copies)
        viewed_shard = self._restore_shard(shard_path, view=True)
        assert viewed_shard is not None
        self._viewed_indexes.append(viewed_shard)
        view_shards = self._view_shards
        if view_shards is None:
            view_shards = Indexes()
            self._view_shards = view_shards
        view_shards.merge(viewed_shard)

        # Create new active shard
        self._active_shard = self._create_shard()

    def _empty_results(self, vectors: NDArray[Any], count: int, is_single: bool) -> Matches | BatchMatches:
        """Return empty results in the appropriate format."""
        if is_single:
            return Matches(
                keys=np.array([], dtype=np.uint64),
                distances=np.array([], dtype=np.float32),
            )
        else:
            num_queries = vectors.shape[0]
            return BatchMatches(
                keys=np.zeros((num_queries, count), dtype=np.uint64),
                distances=np.full((num_queries, count), np.inf, dtype=np.float32),
                counts=np.zeros(num_queries, dtype=np.int64),
            )

    def _merge_search_results(
        self,
        results: list[Matches | BatchMatches],
        count: int,
        radius: float,
        is_single: bool,
    ) -> Matches | BatchMatches:
        """Merge search results from multiple sources, keeping top-k by distance."""
        if is_single:
            return self._merge_single_matches(cast(Sequence[Matches], results), count, radius)
        else:
            return self._merge_batch_matches(cast(Sequence[BatchMatches], results), count, radius)

    def _merge_single_matches(self, matches_list: Sequence[Matches], count: int, radius: float) -> Matches:
        """Merge multiple Matches objects into one, sorted by distance."""
        all_keys = np.concatenate([m.keys for m in matches_list])
        all_distances = np.concatenate([m.distances for m in matches_list])

        # Sort by distance and take top count within radius
        sorted_indices = np.argsort(all_distances)
        if radius < float("inf"):
            mask = all_distances[sorted_indices] <= radius
            sorted_indices = sorted_indices[mask]
        sorted_indices = sorted_indices[:count]

        return Matches(
            keys=all_keys[sorted_indices],
            distances=all_distances[sorted_indices],
            visited_members=sum(m.visited_members for m in matches_list),
            computed_distances=sum(m.computed_distances for m in matches_list),
        )

    def _merge_batch_matches(
        self,
        batch_list: Sequence[BatchMatches],
        count: int,
        radius: float,
    ) -> BatchMatches:
        """Merge multiple BatchMatches objects into one."""
        num_queries = len(batch_list[0])

        merged_keys = []
        merged_distances = []
        merged_counts = []

        for query_idx in range(num_queries):
            # Gather results for this query from all shards
            all_keys = []
            all_distances = []

            for batch in batch_list:
                query_matches = batch[query_idx]
                all_keys.extend(query_matches.keys)
                all_distances.extend(query_matches.distances)

            # Sort and truncate within radius
            all_keys = np.array(all_keys, dtype=np.uint64)
            all_distances = np.array(all_distances, dtype=np.float32)
            sorted_indices = np.argsort(all_distances)
            if radius < float("inf"):
                mask = all_distances[sorted_indices] <= radius
                sorted_indices = sorted_indices[mask]
            sorted_indices = sorted_indices[:count]

            merged_keys.append(all_keys[sorted_indices])
            merged_distances.append(all_distances[sorted_indices])
            merged_counts.append(len(sorted_indices))

        # Pad to uniform shape
        keys_array = np.zeros((num_queries, count), dtype=np.uint64)
        distances_array = np.full((num_queries, count), np.inf, dtype=np.float32)

        for i in range(num_queries):
            n = len(merged_keys[i])
            keys_array[i, :n] = merged_keys[i]
            distances_array[i, :n] = merged_distances[i]

        return BatchMatches(
            keys=keys_array,
            distances=distances_array,
            counts=np.array(merged_counts, dtype=np.int64),
            visited_members=sum(b.visited_members for b in batch_list),
            computed_distances=sum(b.computed_distances for b in batch_list),
        )

    def _apply_radius_filter(
        self,
        result: Matches | BatchMatches,
        radius: float,
        is_single: bool,
    ) -> Matches | BatchMatches:
        """Filter search results to only include matches within radius."""
        if is_single:
            matches = cast(Matches, result)
            mask = matches.distances <= radius
            return Matches(
                keys=matches.keys[mask],
                distances=matches.distances[mask],
                visited_members=matches.visited_members,
                computed_distances=matches.computed_distances,
            )
        else:
            batch = cast(BatchMatches, result)
            num_queries = len(batch)
            count = batch.keys.shape[1]
            keys_array = np.zeros((num_queries, count), dtype=np.uint64)
            distances_array = np.full((num_queries, count), np.inf, dtype=np.float32)
            counts = []

            for i in range(num_queries):
                mask = batch.distances[i] <= radius
                filtered_keys = batch.keys[i][mask]
                filtered_dists = batch.distances[i][mask]
                n = len(filtered_keys)
                keys_array[i, :n] = filtered_keys
                distances_array[i, :n] = filtered_dists
                counts.append(n)

            return BatchMatches(
                keys=keys_array,
                distances=distances_array,
                counts=np.array(counts, dtype=np.int64),
                visited_members=batch.visited_members,
                computed_distances=batch.computed_distances,
            )

    def __repr__(self) -> str:
        """Return string representation of the sharded index."""
        return f"ShardedIndex({self.size} vectors in {self.shard_count} shards, ndim={self.ndim}, path={self._path})"
