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

from iscc_usearch.bloom import ScalableBloomFilter
from iscc_usearch.utils import timer

__all__ = ["ShardedIndex", "ShardedIndex128", "ShardedIndexedKeys", "ShardedIndexedVectors"]

# Default bloom filter file name
BLOOM_FILENAME = "bloom.isbf"

# Default shard size: 1GB
DEFAULT_SHARD_SIZE = 1024 * 1024 * 1024

# NumPy dtype for 128-bit UUID keys
UUID_DTYPE = np.dtype("V16")

# Mapping from ScalarKind to numpy dtype
SCALAR_KIND_TO_NUMPY_DTYPE: dict[ScalarKind, np.dtype] = {
    ScalarKind.B1: np.dtype(np.uint8),
    ScalarKind.F16: np.dtype(np.float16),
    ScalarKind.F32: np.dtype(np.float32),
    ScalarKind.F64: np.dtype(np.float64),
    ScalarKind.I8: np.dtype(np.int8),
    ScalarKind.I16: np.dtype(np.int16),
    ScalarKind.I32: np.dtype(np.int32),
    ScalarKind.I64: np.dtype(np.int64),
    ScalarKind.U8: np.dtype(np.uint8),
    ScalarKind.U16: np.dtype(np.uint16),
    ScalarKind.U32: np.dtype(np.uint32),
    ScalarKind.U64: np.dtype(np.uint64),
}


def _vector_width(ndim: int, dtype: ScalarKind) -> int:
    """Compute the vector width (number of array elements) for a given ndim and dtype.

    For ScalarKind.B1, ndim is the number of bits but vectors are stored as packed
    uint8 bytes, so the width is ceil(ndim/8). For all other types, width equals ndim.

    :param ndim: Number of dimensions (bits for B1, elements for others)
    :param dtype: The scalar kind of the index
    :return: Number of elements in the vector array
    """
    if dtype == ScalarKind.B1:
        return (ndim + 7) // 8
    return ndim


class ShardedIndexedKeys:
    """Lazy key iterator across all shards.

    Provides memory-efficient access to keys without materializing them all at once.
    Supports iteration, length, indexing, slicing, and numpy array conversion.

    This is a live view - it reflects the current state of the index at iteration time.
    """

    def __init__(self, sharded_index: "ShardedIndex") -> None:
        """Initialize with reference to sharded index.

        :param sharded_index: The ShardedIndex to iterate keys from
        """
        self._index = sharded_index

    def __len__(self) -> int:
        """Return total number of keys across all shards."""
        return len(self._index)

    def __iter__(self):
        """Yield keys from all shards lazily.

        Iterates through view shards first, then active shard.
        """
        for idx in self._index._viewed_indexes:
            yield from idx.keys
        if self._index._active_shard is not None:
            yield from self._index._active_shard.keys

    def __getitem__(self, index: int | slice) -> int | NDArray[np.uint64]:
        """Support indexing and slicing.

        :param index: Integer index or slice
        :return: Single key or array of keys
        """
        if isinstance(index, slice):
            # Handle slicing by materializing the requested range
            arr = np.asarray(self)
            return arr[index]
        else:
            # Handle single index
            if index < 0:
                index = len(self) + index
            if index < 0 or index >= len(self):
                raise IndexError("index out of range")

            # Iterate through shards to find the key at the given index
            current = 0
            for idx in self._index._viewed_indexes:
                shard_len = len(idx)
                if current + shard_len > index:
                    return idx.keys[index - current]
                current += shard_len

            if self._index._active_shard is not None:
                return self._index._active_shard.keys[index - current]

            raise IndexError("index out of range")  # pragma: no cover

    def __array__(self, dtype: Any = None) -> NDArray:
        """Convert to numpy array by concatenating shard key arrays.

        :param dtype: Optional dtype override
        :return: Numpy array of all keys
        """
        arrays = []
        for idx in self._index._viewed_indexes:
            if len(idx) > 0:  # pragma: no branch - viewed shards are never empty
                arrays.append(np.asarray(idx.keys))
        if self._index._active_shard is not None and len(self._index._active_shard) > 0:
            arrays.append(np.asarray(self._index._active_shard.keys))
        if not arrays:
            result = np.array([], dtype=self._index._key_dtype)
        else:
            result = np.concatenate(arrays)
        if dtype is not None and dtype != result.dtype:
            return result.astype(dtype)
        return result

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ShardedIndexedKeys(count={len(self)})"


class ShardedIndexedVectors:
    """Lazy vector iterator across all shards.

    Provides memory-efficient access to vectors without materializing them all at once.
    Supports iteration, length, indexing, slicing, and numpy array conversion.

    This is a live view - it reflects the current state of the index at iteration time.

    Note: Unlike usearch Index.vectors which returns an np.ndarray immediately,
    this returns a lazy iterator appropriate for larger-than-RAM indexes.
    """

    def __init__(self, sharded_index: "ShardedIndex") -> None:
        """Initialize with reference to sharded index.

        :param sharded_index: The ShardedIndex to iterate vectors from
        """
        self._index = sharded_index

    def __len__(self) -> int:
        """Return total number of vectors across all shards."""
        return len(self._index)

    def __iter__(self):
        """Yield vectors from all shards lazily.

        Iterates through view shards first, then active shard.
        """
        for idx in self._index._viewed_indexes:
            yield from idx.vectors
        if self._index._active_shard is not None:
            yield from self._index._active_shard.vectors

    def __getitem__(self, index: int | slice) -> NDArray[Any]:
        """Support indexing and slicing.

        :param index: Integer index or slice
        :return: Single vector or array of vectors
        """
        if isinstance(index, slice):
            # Handle slicing by materializing the requested range
            vectors = list(self)
            sliced = vectors[index]
            if not sliced:
                npdtype = SCALAR_KIND_TO_NUMPY_DTYPE.get(self._index.dtype, np.dtype(np.uint8))
                width = _vector_width(self._index.ndim, self._index.dtype)
                return np.empty((0, width), dtype=npdtype)
            return np.vstack(sliced)
        else:
            # Handle single index
            if index < 0:
                index = len(self) + index
            if index < 0 or index >= len(self):
                raise IndexError("index out of range")

            # Iterate through shards to find the vector at the given index
            current = 0
            for idx in self._index._viewed_indexes:
                shard_len = len(idx)
                if current + shard_len > index:
                    return idx.vectors[index - current]
                current += shard_len

            if self._index._active_shard is not None:
                return self._index._active_shard.vectors[index - current]

            raise IndexError("index out of range")  # pragma: no cover

    def __array__(self, dtype: Any = None) -> NDArray[Any]:
        """Support numpy array conversion.

        Warning: This materializes all vectors into memory.

        :param dtype: Optional dtype for the result array
        :return: 2D numpy array of all vectors
        """
        vectors = list(self)
        if not vectors:
            default_dtype = SCALAR_KIND_TO_NUMPY_DTYPE.get(self._index.dtype, np.dtype(np.uint8))
            width = _vector_width(self._index.ndim, self._index.dtype)
            return np.empty((0, width), dtype=dtype or default_dtype)
        result = np.vstack(vectors)
        if dtype is not None and result.dtype != dtype:
            return result.astype(dtype)
        return result

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ShardedIndexedVectors(count={len(self)})"


class ShardedIndex:
    """Sharded vector index for scalable append-only storage.

    Wraps usearch Index/Indexes to provide automatic sharding when the active
    shard exceeds the configured size limit. Finished shards are memory-mapped
    (view mode) for efficient read-only access, while the active shard is
    fully loaded (load mode) for read-write operations.

    CONCURRENCY: Single-process only. No file locking. Use async/await within
    a single process for concurrent access.

    :param ndim: Number of vector dimensions (auto-detected from existing shards if omitted)
    :param metric: Distance metric (MetricKind or CompiledMetric)
    :param dtype: Scalar type for vectors (ScalarKind)
    :param connectivity: HNSW connectivity parameter (M)
    :param expansion_add: Search depth on insertions (efConstruction)
    :param expansion_search: Search depth on queries (ef)
    :param multi: Allow multiple vectors per key
    :param path: Directory path for shard storage (required)
    :param shard_size: Size limit in bytes before rotating shards (default 1GB)
    :param bloom_filter: Enable bloom filter for fast non-existent key rejection
    """

    def __init__(
        self,
        *,
        ndim: int | None = None,
        metric: MetricKind | Any = MetricKind.Cos,
        dtype: ScalarKind | str | None = None,
        connectivity: int | None = None,
        expansion_add: int | None = None,
        expansion_search: int | None = None,
        multi: bool = False,
        path: str | os.PathLike,
        shard_size: int = DEFAULT_SHARD_SIZE,
        bloom_filter: bool = True,
    ) -> None:
        """Initialize a sharded index."""
        self._path = Path(path)
        self._shard_size = shard_size
        self._use_bloom = bloom_filter

        # Initialize shard containers early (used by _discover_shards)
        self._view_shards: Indexes | None = None
        self._active_shard: Index | None = None
        self._active_shard_path: Path | None = None
        self._viewed_indexes: list[Index] = []
        self._cached_shards: list[Path] | None = None

        # Initialize bloom filter (loaded/created in load/view)
        self._bloom: ScalableBloomFilter | None = None

        # Amortized rotation check: skip O(n) serialized_length on every add
        self._adds_until_size_check: int = 0

        # Create directory if needed
        self._path.mkdir(parents=True, exist_ok=True)

        # Discover existing shards
        existing_shards = self._discover_shards()

        # Auto-detect ndim from existing shards if not provided
        ndim, metric, dtype = self._resolve_config(ndim, metric, dtype, existing_shards)

        # Store config for creating new shards
        self._config: dict[str, Any] = {
            "ndim": ndim,
            "metric": metric,
            "dtype": dtype,
            "connectivity": connectivity,
            "expansion_add": expansion_add,
            "expansion_search": expansion_search,
            "multi": multi,
        }

        if existing_shards:
            # Load existing index
            self._load_existing()
        else:
            # Create first active shard and bloom filter for new index
            self._active_shard = self._create_shard()
            if self._use_bloom:
                self._bloom = ScalableBloomFilter()

    # --- Key handling hooks (override in _UuidKeyMixin for uuid keys) ---

    def _is_single_key(self, keys: Any) -> bool:
        """Check if keys represents a single key (vs a batch)."""
        return isinstance(keys, int)

    def _bloom_key(self, key: Any) -> int:
        """Convert a single key to bloom filter format."""
        return int(key)

    def _bloom_keys(self, keys_arr: NDArray) -> list[int]:
        """Convert a key array to bloom filter batch format."""
        return keys_arr.tolist()

    def _normalize_batch_keys(self, keys: Any) -> NDArray:
        """Normalize batch keys to the appropriate numpy array type."""
        return np.asarray(keys, dtype=np.uint64)

    def _shard_batch_keys(self, keys_arr: NDArray) -> Any:
        """Convert batch key array to format for usearch shard operations."""
        return keys_arr.tolist()

    @property
    def _key_dtype(self) -> np.dtype:
        """NumPy dtype for keys."""
        return np.dtype(np.uint64)

    # --- View shard hooks ---

    def _register_view_shard(self, shard: Index) -> None:
        """Register a shard in view mode for read-only access.

        Adds to both _viewed_indexes list and _view_shards (Indexes) container.
        """
        self._viewed_indexes.append(shard)
        if self._view_shards is None:
            self._view_shards = Indexes()
        self._view_shards.merge(shard)

    def _search_view_shards(
        self,
        vectors: NDArray[Any],
        count: int,
        threads: int,
        exact: bool,
        progress: Callable[[int, int], bool] | None,
    ) -> Matches | BatchMatches | None:
        """Search across all view shards.

        :return: Search results or None if no view shards exist.
        """
        if self._view_shards is not None and len(self._view_shards) > 0:
            return self._view_shards.search(vectors, count=count, threads=threads, exact=exact, progress=progress)
        return None

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
        """
        if self._active_shard is None:
            self._active_shard = self._create_shard()

        # Delegate to active shard
        result = self._active_shard.add(keys, vectors, copy=copy, threads=threads, log=log, progress=progress)

        # Always update bloom filter if it exists (keeps it in sync regardless of _use_bloom)
        # Note: usearch always returns keys as numpy array, even for single adds
        if self._bloom is not None:
            for key in np.atleast_1d(result):
                self._bloom.add(self._bloom_key(key))

        # Amortized rotation check: only compute serialized_length when countdown expires
        count_added = len(np.atleast_1d(result))
        self._adds_until_size_check -= count_added
        if self._adds_until_size_check <= 0:
            size = self._active_shard.serialized_length
            if size > self._shard_size:
                self._rotate_shard()
            else:
                self._schedule_next_size_check(size)

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
            raise ValueError("`count` must be >= 1")

        vectors = np.asarray(vectors)
        is_single = vectors.ndim == 1

        view_results: Matches | BatchMatches | None = None
        active_results: Matches | BatchMatches | None = None

        # Search view shards (via hook - Indexes or per-shard iteration)
        view_results = self._search_view_shards(vectors, count, threads, exact, progress)

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

        # Fast paths - avoid list allocation for common cases
        if view_results is None and active_results is None:
            return self._empty_results(vectors, count, is_single)

        if view_results is None:
            return active_results  # type: ignore[return-value]

        if active_results is None:
            # Apply radius filter (Indexes.search doesn't support radius)
            if radius < float("inf"):
                return self._apply_radius_filter(view_results, radius, is_single)
            return view_results

        # Merge results from both sources (applies radius filter)
        return self._merge_search_results([view_results, active_results], count, radius, is_single)

    def get(
        self,
        keys: int | Any,
        dtype: Any = None,
    ) -> NDArray[Any] | list | None:
        """Retrieve vectors by key from any shard.

        :param keys: Integer key(s) to lookup
        :param dtype: Optional data type for returned vectors
        :return: Vector(s) or None for missing keys
        """
        if self._is_single_key(keys):
            return self._get_single(keys, dtype)
        return self._get_batch(keys, dtype)

    def _get_single(self, key: int, dtype: Any = None) -> NDArray[Any] | None:
        """Get a single vector by key from any shard."""
        # Fast path: bloom filter says definitely not present
        if self._use_bloom and self._bloom is not None:
            if not self._bloom.contains(key):
                return None

        # Check active shard first
        if self._active_shard is not None:
            if self._active_shard.contains(key):
                return self._active_shard.get(key, dtype=dtype)

        # Check view shards
        for idx in self._viewed_indexes:
            if idx.contains(key):
                return idx.get(key, dtype=dtype)

        return None

    def _get_batch(self, keys: Any, dtype: Any = None) -> list:
        """Get multiple vectors by keys from any shard."""
        keys_arr = self._normalize_batch_keys(keys)
        n = len(keys_arr)
        if n == 0:
            return []

        results: list[NDArray[Any] | None] = [None] * n
        found = np.zeros(n, dtype=bool)

        # Fast path: use bloom filter to mark definitely-not-present keys as "found" (None)
        if self._use_bloom and self._bloom is not None:
            bloom_results = self._bloom.contains_batch(self._bloom_keys(keys_arr))
            # Keys that bloom says definitely don't exist - mark as "done" (result stays None)
            for i, maybe_present in enumerate(bloom_results):
                if not maybe_present:
                    found[i] = True  # Skip this key in shard search

        def process_shard(idx: Index) -> None:
            """Process a single shard, updating results for unfound keys."""
            nonlocal found
            unfound_mask = ~found
            unfound_indices = np.where(unfound_mask)[0]
            unfound_keys = keys_arr[unfound_indices]

            # Check which keys exist in this shard
            exists = np.asarray(idx.contains(self._shard_batch_keys(unfound_keys)), dtype=bool)
            if not exists.any():
                return

            # Get vectors for existing keys (batch operation)
            exist_local_indices = np.where(exists)[0]
            exist_keys = unfound_keys[exist_local_indices]
            exist_orig_indices = unfound_indices[exist_local_indices]

            vectors = idx.get(self._shard_batch_keys(exist_keys), dtype=dtype)

            # Store results
            for orig_idx, vec in zip(exist_orig_indices, vectors):
                results[orig_idx] = vec
                found[orig_idx] = True

        # Process active shard first
        if self._active_shard is not None:
            process_shard(self._active_shard)

        # Process view shards
        for idx in self._viewed_indexes:
            if found.all():
                break
            process_shard(idx)

        return results

    def contains(self, keys: int | Any) -> bool | NDArray[np.bool_]:
        """Check if keys exist in any shard.

        When bloom_filter=True (default), uses bloom filter to quickly reject non-existent keys.

        :param keys: Integer key(s) to check
        :return: Boolean or array of booleans
        """
        if self._is_single_key(keys):
            return self._contains_single(keys)
        return self._contains_batch(keys)

    def _contains_single(self, key: int) -> bool:
        """Check if a single key exists in any shard."""
        # Fast path: bloom filter says definitely not present
        if self._use_bloom and self._bloom is not None:
            if not self._bloom.contains(key):
                return False

        # Check active shard first (most likely location for recent keys)
        if self._active_shard is not None:
            if self._active_shard.contains(key):
                return True

        # Check view shards
        for idx in self._viewed_indexes:
            if idx.contains(key):
                return True

        return False

    def _contains_batch(self, keys: Any) -> NDArray[np.bool_]:
        """Check if multiple keys exist in any shard (OR aggregation)."""
        keys_arr = self._normalize_batch_keys(keys)
        if len(keys_arr) == 0:
            return np.array([], dtype=bool)

        result = np.zeros(len(keys_arr), dtype=bool)

        # Fast path: use bloom filter to identify definitely-not-present keys
        if self._use_bloom and self._bloom is not None:
            # Check which keys might exist (bloom says "maybe")
            bloom_results = self._bloom.contains_batch(self._bloom_keys(keys_arr))
            maybe_present = np.array(bloom_results, dtype=bool)

            # If all keys are definitely not present, return early
            if not maybe_present.any():
                return result

            # Only check shards for keys that might exist
            keys_to_check = keys_arr[maybe_present]
            indices_to_check = np.where(maybe_present)[0]
        else:
            keys_to_check = keys_arr
            indices_to_check = np.arange(len(keys_arr))

        partial_result = np.zeros(len(keys_to_check), dtype=bool)

        # Check active shard
        if self._active_shard is not None:
            active_result = self._active_shard.contains(self._shard_batch_keys(keys_to_check))
            partial_result |= np.asarray(active_result, dtype=bool)

        # Check view shards
        for idx in self._viewed_indexes:
            if partial_result.all():
                break  # Short-circuit: all keys found
            shard_result = idx.contains(self._shard_batch_keys(keys_to_check))
            partial_result |= np.asarray(shard_result, dtype=bool)

        # Map partial results back to full result array
        result[indices_to_check] = partial_result

        return result

    def __contains__(self, keys: int | Any) -> bool | NDArray[np.bool_]:
        """Support 'in' operator."""
        return self.contains(keys)

    def count(self, keys: int | Any) -> int | NDArray[np.uint64]:
        """Count occurrences of keys across all shards (sum aggregation).

        :param keys: Integer key(s) to count
        :return: Count or array of counts
        """
        if self._is_single_key(keys):
            return self._count_single(keys)
        return self._count_batch(keys)

    def _count_single(self, key: int) -> int:
        """Count occurrences of a single key across all shards."""
        # Fast path: bloom filter says definitely not present
        if self._use_bloom and self._bloom is not None:
            if not self._bloom.contains(key):
                return 0

        total = 0

        if self._active_shard is not None:
            total += self._active_shard.count(key)

        for idx in self._viewed_indexes:
            total += idx.count(key)

        return total

    def _count_batch(self, keys: Any) -> NDArray[np.uint64]:
        """Count occurrences of multiple keys across all shards (sum)."""
        keys_arr = self._normalize_batch_keys(keys)
        if len(keys_arr) == 0:
            return np.array([], dtype=np.uint64)

        total = np.zeros(len(keys_arr), dtype=np.uint64)

        # Fast path: use bloom filter to skip definitely-not-present keys
        if self._use_bloom and self._bloom is not None:
            bloom_results = self._bloom.contains_batch(self._bloom_keys(keys_arr))
            maybe_present = np.array(bloom_results, dtype=bool)
            if not maybe_present.any():
                return total
            keys_to_count = keys_arr[maybe_present]
            indices_to_count = np.where(maybe_present)[0]
        else:
            keys_to_count = keys_arr
            indices_to_count = np.arange(len(keys_arr))

        partial_total = np.zeros(len(keys_to_count), dtype=np.uint64)

        if self._active_shard is not None:
            partial_total += np.asarray(
                self._active_shard.count(self._shard_batch_keys(keys_to_count)), dtype=np.uint64
            )

        for idx in self._viewed_indexes:
            partial_total += np.asarray(idx.count(self._shard_batch_keys(keys_to_count)), dtype=np.uint64)

        total[indices_to_count] = partial_total
        return total

    # === Persistence ===

    def save(
        self,
        path_or_buffer: str | os.PathLike | None = None,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None:
        """Save active shard and bloom filter to disk.

        :param path_or_buffer: Ignored (uses internal path management)
        :param progress: Progress callback
        """
        # Save bloom filter if it exists
        if self._bloom is not None:
            bloom_path = self._path / BLOOM_FILENAME
            with timer("ShardedIndex save bloom filter"):
                self._bloom.save(bloom_path)

        if self._active_shard is None or len(self._active_shard) == 0:
            return

        shard_path = self._get_active_shard_path()
        with timer(f"ShardedIndex save {shard_path.name}"):
            self._active_shard.save(shard_path, progress=progress)
        self._active_shard_path = shard_path
        # Invalidate cache since new shard file may have been created
        self._invalidate_shard_cache()

    def rebuild_bloom(self, save: bool = True, log_progress: bool = True) -> int:
        """Rebuild bloom filter from all existing keys.

        Use this to populate the bloom filter for an existing index that was
        created without bloom filter support, or to repair a corrupted filter.

        Processes keys shard-by-shard in batches for efficiency.

        :param save: Whether to save the bloom filter to disk after rebuilding
        :param log_progress: Whether to log progress per shard
        :return: Number of keys added to the bloom filter
        """
        from loguru import logger

        # Create fresh bloom filter
        self._bloom = ScalableBloomFilter()
        self._use_bloom = True

        total = len(self)
        if log_progress:
            logger.info(f"Rebuilding bloom filter for {total:,} keys...")

        # Process each shard's keys as a batch (much faster than one-by-one)
        count = 0
        all_shards = self._viewed_indexes + ([self._active_shard] if self._active_shard else [])
        num_shards = len(all_shards)

        for i, idx in enumerate(all_shards):
            if idx is None or len(idx) == 0:  # pragma: no cover
                continue

            # Get all keys from this shard as numpy array
            shard_keys = np.asarray(idx.keys, dtype=self._key_dtype)
            shard_count = len(shard_keys)

            # Add to bloom filter using proper batch operation (handles capacity/growth)
            self._bloom.add_batch(self._bloom_keys(shard_keys))
            count += shard_count

            if log_progress:
                logger.info(
                    f"  Shard {i + 1}/{num_shards}: +{shard_count:,} keys "
                    f"(total: {count:,}, filters: {self._bloom.filter_count})"
                )

        if log_progress:
            logger.info(f"Bloom filter rebuilt with {count:,} keys")

        # Save if requested
        if save:
            bloom_path = self._path / BLOOM_FILENAME
            with timer("ShardedIndex save bloom filter"):
                self._bloom.save(bloom_path)

        return count

    def _load_existing(self) -> None:
        """Load existing shards from directory.

        Finished shards are memory-mapped (read-only), last shard is loaded for writes.
        """
        self._invalidate_shard_cache()
        shard_files = self._discover_shards()

        # Always load bloom filter if file exists to keep it in sync
        # _use_bloom only controls whether to USE it for fast rejection in lookups
        self._bloom = self._load_bloom_if_exists()

        if not shard_files:
            self._active_shard = self._create_shard()
            self._view_shards = None
            # Create bloom filter for new empty index if enabled
            # (Skip if bloom file exists but shards don't - corruption recovery)
            if self._use_bloom and self._bloom is None:
                self._bloom = ScalableBloomFilter()
            return  # pragma: no cover - defensive path for corruption recovery

        with timer(f"ShardedIndex load {len(shard_files)} shards from {self._path}", log_start=True):
            # All but the last shard go into view mode
            view_paths = shard_files[:-1]
            active_path = shard_files[-1]

            # Restore view shards individually (workaround for usearch bug #643:
            # Indexes(paths=[...]) segfaults). References kept to prevent GC.
            self._viewed_indexes = []
            self._view_shards = None
            for p in view_paths:
                viewed = self._restore_shard(p, view=True)
                if viewed is None:  # pragma: no cover
                    raise RuntimeError(f"Failed to restore shard: {p}")
                self._register_view_shard(viewed)

            # Load active shard (writable) and track its path for save()
            active_shard = self._restore_shard(active_path, view=False)
            if active_shard is None:  # pragma: no cover
                raise RuntimeError(f"Failed to restore shard: {active_path}")
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
        total = sum(len(idx) for idx in self._viewed_indexes)
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
        """Serialized length of active shard."""
        if self._active_shard is not None:
            return self._active_shard.serialized_length
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
    def keys(self) -> ShardedIndexedKeys:
        """Lazy iterator over all keys across all shards.

        Returns a ShardedIndexedKeys object that supports:
        - Iteration: for key in idx.keys
        - Length: len(idx.keys)
        - Indexing: idx.keys[0], idx.keys[-1]
        - Slicing: idx.keys[:10]
        - Numpy conversion: np.asarray(idx.keys)

        This is a live view - reflects current state at iteration time.

        :return: ShardedIndexedKeys iterator
        """
        return ShardedIndexedKeys(self)

    @property
    def vectors(self) -> ShardedIndexedVectors:
        """Lazy iterator over all vectors across all shards.

        Returns a ShardedIndexedVectors object that supports:
        - Iteration: for vec in idx.vectors
        - Length: len(idx.vectors)
        - Indexing: idx.vectors[0], idx.vectors[-1]
        - Slicing: idx.vectors[:10]
        - Numpy conversion: np.asarray(idx.vectors)

        This is a live view - reflects current state at iteration time.

        Note: Unlike usearch Index.vectors which returns an np.ndarray immediately,
        this returns a lazy iterator appropriate for larger-than-RAM indexes.

        :return: ShardedIndexedVectors iterator
        """
        return ShardedIndexedVectors(self)

    # === Helper Methods ===

    def _resolve_config(
        self,
        ndim: int | None,
        metric: MetricKind | Any,
        dtype: ScalarKind | str | None,
        existing_shards: list[Path],
    ) -> tuple[int, MetricKind | Any, ScalarKind | str | None]:
        """Resolve ndim/metric/dtype from existing shards or validate provided values.

        :param ndim: Provided ndim or None for auto-detection
        :param metric: Provided metric
        :param dtype: Provided dtype
        :param existing_shards: List of existing shard paths
        :return: Resolved (ndim, metric, dtype) tuple
        :raises ValueError: If ndim is None and no existing shards found
        """
        if not existing_shards:
            if ndim is None:
                raise ValueError("ndim is required when creating a new index (no existing shards found)")
            return ndim, metric, dtype

        # Read metadata from first shard
        meta = Index.metadata(str(existing_shards[0]))
        if meta is None:  # pragma: no cover - shard files are always valid in practice
            if ndim is None:
                raise ValueError("ndim is required (failed to read shard metadata)")
            return ndim, metric, dtype

        # Auto-detect from existing shards if not provided
        resolved_ndim = ndim if ndim is not None else meta["dimensions"]
        resolved_dtype = dtype if dtype is not None else meta["kind_scalar"]
        resolved_metric = metric

        return resolved_ndim, resolved_metric, resolved_dtype

    def _create_shard(self) -> Index:
        """Create a new shard. Override in subclasses for custom shard types."""
        return Index(**self._config)

    def _load_bloom_if_exists(self) -> ScalableBloomFilter | None:
        """Load bloom filter from disk if file exists, otherwise return None.

        Always loads the bloom filter regardless of _use_bloom setting to keep it
        in sync with index contents. The _use_bloom flag only controls whether
        to USE the bloom filter for fast rejection in lookups.

        Returns None when no bloom file exists. Call rebuild_bloom() to create
        a bloom filter for existing indexes that don't have one.
        """
        bloom_path = self._path / BLOOM_FILENAME
        if bloom_path.exists():
            with timer("ShardedIndex load bloom filter"):
                return ScalableBloomFilter.load(bloom_path)
        return None

    def _restore_shard(self, path: Path, view: bool) -> Index | None:
        """Restore a shard from disk. Override in subclasses for custom shard types."""
        meta = Index.metadata(str(path))
        if meta is None:  # pragma: no cover - shard files are always valid in practice
            return None
        idx = Index(
            ndim=meta["dimensions"],
            metric=meta["kind_metric"],
            dtype=meta["kind_scalar"],
        )
        if view:
            idx.view(str(path))
        else:
            idx.load(str(path))
        return idx

    def _discover_shards(self) -> list[Path]:
        """Return sorted list of shard_*.usearch files (cached)."""
        if self._cached_shards is None:
            shards = list(self._path.glob("shard_*.usearch"))
            self._cached_shards = sorted(shards, key=lambda p: int(p.stem.split("_")[1]))
        return self._cached_shards

    def _invalidate_shard_cache(self) -> None:
        """Invalidate cached shard list (call after rotation/load/view)."""
        self._cached_shards = None

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

    def _schedule_next_size_check(self, current_size: int) -> None:
        """Estimate how many adds until shard reaches size limit.

        Uses bytes-per-vector to predict remaining capacity, then schedules
        the next check at the halfway point for a safety margin.

        :param current_size: Current serialized_length of the active shard.
        """
        n_vectors = len(self._active_shard) if self._active_shard else 0
        if n_vectors > 0:
            bytes_per_vector = current_size / n_vectors
            remaining_bytes = self._shard_size - current_size
            estimated_remaining = int(remaining_bytes / bytes_per_vector)
            # Check again at halfway to threshold, minimum 1
            self._adds_until_size_check = max(1, estimated_remaining // 2)
        else:
            self._adds_until_size_check = 1

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
        # Invalidate cache since new shard file was created
        self._invalidate_shard_cache()

        # Load the saved shard in view mode and register it
        viewed_shard = self._restore_shard(shard_path, view=True)
        if viewed_shard is None:  # pragma: no cover
            raise RuntimeError(f"Failed to restore shard: {shard_path}")
        self._register_view_shard(viewed_shard)

        # Create new active shard and reset size check countdown
        self._active_shard = self._create_shard()
        self._adds_until_size_check = 0

    def _empty_results(self, vectors: NDArray[Any], count: int, is_single: bool) -> Matches | BatchMatches:
        """Return empty results in the appropriate format."""
        if is_single:
            return Matches(
                keys=np.array([], dtype=self._key_dtype),
                distances=np.array([], dtype=np.float32),
            )
        else:
            num_queries = vectors.shape[0]
            return BatchMatches(
                keys=np.zeros((num_queries, count), dtype=self._key_dtype),
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
        # Fast path for common two-source case (avoids generator overhead)
        if len(matches_list) == 2:
            all_keys = np.concatenate((matches_list[0].keys, matches_list[1].keys))
            all_distances = np.concatenate((matches_list[0].distances, matches_list[1].distances))
        else:
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
        """Merge multiple BatchMatches objects into one (vectorized)."""
        num_queries = len(batch_list[0])

        # Stack all results: (num_queries, total_count_across_batches)
        all_keys = np.concatenate([b.keys for b in batch_list], axis=1)
        all_distances = np.concatenate([b.distances for b in batch_list], axis=1)

        # Argsort each row independently
        sorted_indices = np.argsort(all_distances, axis=1)

        # Gather sorted values using advanced indexing
        row_idx = np.arange(num_queries)[:, None]
        sorted_keys = all_keys[row_idx, sorted_indices]
        sorted_distances = all_distances[row_idx, sorted_indices]

        # Apply radius filter and compute counts
        if radius < float("inf"):
            valid_mask = sorted_distances <= radius
            # Count valid entries per row, capped at count
            counts = np.minimum(valid_mask.sum(axis=1), count)
            # Invalidate entries beyond radius (dtype-safe zero-fill for V16 compat)
            result_keys = np.zeros_like(sorted_keys)
            result_keys[valid_mask] = sorted_keys[valid_mask]
            sorted_keys = result_keys
            sorted_distances = np.where(valid_mask, sorted_distances, np.inf)
        else:
            # Count non-inf entries per row, capped at count
            counts = np.minimum((sorted_distances < np.inf).sum(axis=1), count)

        # Truncate to requested count
        keys_array = sorted_keys[:, :count].copy()
        distances_array = sorted_distances[:, :count].copy()

        return BatchMatches(
            keys=keys_array,
            distances=distances_array,
            counts=counts.astype(np.int64),
            visited_members=sum(b.visited_members for b in batch_list),
            computed_distances=sum(b.computed_distances for b in batch_list),
        )

    def _apply_radius_filter(
        self,
        result: Matches | BatchMatches,
        radius: float,
        is_single: bool,
    ) -> Matches | BatchMatches:
        """Filter search results to only include matches within radius (vectorized)."""
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
            # Vectorized radius mask across all queries
            mask = batch.distances <= radius
            counts = mask.sum(axis=1).astype(np.int64)

            # Apply mask: set invalid entries to 0/inf (dtype-safe zero-fill for V16 compat)
            keys_array = np.zeros_like(batch.keys)
            keys_array[mask] = batch.keys[mask]
            distances_array = np.where(mask, batch.distances, np.inf)

            return BatchMatches(
                keys=keys_array,
                distances=distances_array,
                counts=counts,
                visited_members=batch.visited_members,
                computed_distances=batch.computed_distances,
            )

    def __repr__(self) -> str:
        """Return string representation of the sharded index."""
        return f"ShardedIndex({self.size} vectors in {self.shard_count} shards, ndim={self.ndim}, path={self._path})"


class _UuidKeyMixin:
    """Mixin providing 128-bit UUID key handling hooks and validation.

    Overrides the 6 key-handling hooks on ShardedIndex for bytes(16) / V16 keys,
    adds strict validation on add(), and widens dispatch on get/contains/count
    to catch malformed keys before they reach the bloom filter fast-path.

    Only used via multiple inheritance with ShardedIndex (or its subclasses).
    """

    # Attributes/methods provided by ShardedIndex at runtime (declared for type checking)
    _viewed_indexes: list[Index]
    _merge_search_results: Callable[..., Matches | BatchMatches]

    # --- Key handling hook overrides ---

    def _is_single_key(self, keys: Any) -> bool:
        """Check if keys is a single bytes key."""
        return isinstance(keys, bytes)

    def _bloom_key(self, key: Any) -> bytes:
        """Convert a single key to bloom filter format."""
        return bytes(key)

    def _bloom_keys(self, keys_arr: NDArray) -> list[bytes]:
        """Convert a V16 key array to bloom filter batch format."""
        return [bytes(k) for k in keys_arr]

    def _normalize_batch_keys(self, keys: Any) -> NDArray:
        """Normalize batch keys to V16 numpy array."""
        if isinstance(keys, np.ndarray):
            if keys.dtype == UUID_DTYPE:
                return keys
            raise ValueError(f"Batch keys must have dtype 'V16', got {keys.dtype}")
        try:
            keys_list = list(keys)
        except TypeError:
            raise ValueError(f"UUID keys must be bytes(16) or iterable of bytes(16), got {type(keys).__name__}")
        if keys_list and not all(isinstance(k, bytes) and len(k) == 16 for k in keys_list):
            raise ValueError("All batch keys must be bytes of length 16")
        return np.array(keys_list, dtype=UUID_DTYPE)

    def _shard_batch_keys(self, keys_arr: NDArray) -> Any:
        """Convert batch key array to format for usearch shard operations."""
        return keys_arr  # usearch uuid mode accepts V16 arrays directly

    @property
    def _key_dtype(self) -> np.dtype:
        """NumPy dtype for 128-bit keys."""
        return UUID_DTYPE

    # --- Validation on write path ---

    def add(self, keys: Any, vectors: NDArray[Any], **kwargs: Any) -> Any:
        """Add vectors with strict 128-bit key validation.

        :param keys: bytes(16) key or V16 ndarray batch
        :param vectors: Vector or batch of vectors to add
        :return: Key(s) for added vectors
        :raises ValueError: If keys is None, wrong type, or wrong length
        """
        if keys is None:
            raise ValueError("Auto-key generation not supported for 128-bit keys")
        if isinstance(keys, bytes):
            if len(keys) != 16:
                raise ValueError(f"UUID key must be exactly 16 bytes, got {len(keys)}")
        elif isinstance(keys, np.ndarray):
            if keys.dtype != UUID_DTYPE:
                raise ValueError(f"Batch keys must have dtype 'V16', got {keys.dtype}")
        else:
            raise ValueError("UUID keys must be bytes(16) or np.ndarray with dtype 'V16'")
        return super().add(keys, vectors, **kwargs)  # type: ignore[misc]

    # --- Validation on read paths ---

    def _check_single_key(self, key: Any) -> None:
        """Validate a single bytes key.

        :raises ValueError: If key is not bytes of length 16
        """
        if not isinstance(key, bytes) or len(key) != 16:
            raise ValueError(f"UUID key must be bytes of length 16, got {type(key).__name__}")

    def get(self, keys: Any, dtype: Any = None) -> Any:
        """Retrieve vectors with single-key validation."""
        if self._is_single_key(keys):
            self._check_single_key(keys)
        return super().get(keys, dtype=dtype)  # type: ignore[misc]

    def contains(self, keys: Any) -> Any:
        """Check key existence with single-key validation."""
        if self._is_single_key(keys):
            self._check_single_key(keys)
        return super().contains(keys)  # type: ignore[misc]

    def count(self, keys: Any) -> Any:
        """Count key occurrences with single-key validation."""
        if self._is_single_key(keys):
            self._check_single_key(keys)
        return super().count(keys)  # type: ignore[misc]


class ShardedIndex128(_UuidKeyMixin, ShardedIndex):
    """Sharded vector index with 128-bit UUID keys.

    Uses usearch's key_kind="uuid" for 128-bit composite keys represented as
    bytes(16) for single keys and np.dtype('V16') arrays for batches.

    Auto-generation of keys (keys=None) is not supported — all keys must be
    provided explicitly.
    """

    def __init__(self, *, path: str | os.PathLike, **kwargs: Any) -> None:
        """Initialize a 128-bit sharded index.

        :param path: Directory path for shard storage
        :param kwargs: Passed to ShardedIndex (key_kind is absorbed if present)
        """
        kwargs.pop("key_kind", None)
        super().__init__(path=path, **kwargs)

    def _create_shard(self) -> Index:
        """Create a uuid-keyed shard."""
        return Index(**self._config, key_kind="uuid")

    def _restore_shard(self, path: Path, view: bool) -> Index | None:
        """Restore a uuid-keyed shard from disk."""
        meta = Index.metadata(str(path))
        if meta is None:  # pragma: no cover
            return None
        idx = Index(
            ndim=meta["dimensions"],
            metric=meta["kind_metric"],
            dtype=meta["kind_scalar"],
            key_kind="uuid",
        )
        if view:
            idx.view(str(path))
        else:
            idx.load(str(path))
        return idx
