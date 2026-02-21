"""Sharded NPHD index for scalable variable-length binary bit-vector storage."""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from usearch.index import BatchMatches, Index, Matches

from usearch.index import MetricKind

from iscc_usearch.nphd import pad_vectors, unpad_vectors
from iscc_usearch.sharded import DEFAULT_SHARD_SIZE, ShardedIndex, _UuidKeyMixin

_SENTINEL = object()

__all__ = ["ShardedNphdIndex", "ShardedNphdIndex128", "ShardedNphdIndexedVectors"]


class ShardedNphdIndexedVectors:
    """Lazy vector iterator across all shards returning unpadded vectors.

    Provides memory-efficient access to vectors without materializing them all at once.
    Supports iteration, length, indexing, slicing, and numpy array conversion.

    This is a live view - it reflects the current state of the index at iteration time.
    Vectors are returned unpadded (variable-length), consistent with the get() API.
    """

    def __init__(self, sharded_index: "ShardedNphdIndex") -> None:
        """Initialize with reference to sharded NPHD index.

        :param sharded_index: The ShardedNphdIndex to iterate vectors from
        """
        self._index = sharded_index

    def __len__(self) -> int:
        """Return total number of vectors across all shards."""
        return len(self._index)

    def _needs_filtering(self) -> bool:
        """Check if iteration needs tombstone/dedup filtering."""
        return bool(self._index._tombstones) or self._index._needs_compact

    def __iter__(self):
        """Yield unpadded vectors from all shards lazily.

        Without cross-shard overlap: view shards then active shard (no dedup needed).
        With overlap: active shard first, then view shards newest-to-oldest
        with tombstoned and duplicate keys' vectors excluded.
        """
        tombstones = self._index._tombstones

        # Fast path: no cross-shard overlap — keys are unique across shards
        if not tombstones and not self._index._needs_compact:
            for idx in self._index._viewed_indexes:
                for vec in idx.vectors:
                    yield unpad_vectors(vec.reshape(1, -1))[0]
            if self._index._active_shard is not None:
                for vec in self._index._active_shard.vectors:
                    yield unpad_vectors(vec.reshape(1, -1))[0]
            return

        seen: set = set()

        # Active shard first
        if self._index._active_shard is not None:
            for key, vec in zip(self._index._active_shard.keys, self._index._active_shard.vectors):
                seen.add(self._index._tombstone_key(key))
                yield unpad_vectors(vec.reshape(1, -1))[0]

        # View shards newest-to-oldest
        for idx in reversed(self._index._viewed_indexes):
            for key, vec in zip(idx.keys, idx.vectors):
                canon = self._index._tombstone_key(key)
                if canon not in tombstones and canon not in seen:
                    seen.add(canon)
                    yield unpad_vectors(vec.reshape(1, -1))[0]

    def __getitem__(self, index: int | slice) -> NDArray[np.uint8] | list[NDArray[np.uint8]]:
        """Support indexing and slicing.

        Warning: Slices with active tombstones materialize all vectors into
        memory. Avoid on larger-than-RAM indexes.

        :param index: Integer index or slice
        :return: Single unpadded vector or list of unpadded vectors
        """
        if isinstance(index, slice):
            vectors = list(self)
            return vectors[index]

        # Fast path: no filtering needed - len() is exact
        if not self._needs_filtering():
            if index < 0:
                index = len(self) + index
            if index < 0 or index >= len(self):
                raise IndexError("index out of range")
            current = 0
            for idx in self._index._viewed_indexes:
                shard_len = len(idx)
                if current + shard_len > index:
                    vec = idx.vectors[index - current]
                    return unpad_vectors(vec.reshape(1, -1))[0]
                current += shard_len
            if self._index._active_shard is not None:
                vec = self._index._active_shard.vectors[index - current]
                return unpad_vectors(vec.reshape(1, -1))[0]
            raise IndexError("index out of range")  # pragma: no cover

        # Slow path: dedup/tombstone filtering - len() may overcount
        if index < 0:
            # Two-pass: count exact items first, then islice (avoids full materialization)
            actual_len = sum(1 for _ in self)
            index = actual_len + index
            if index < 0:
                raise IndexError("index out of range")
        result = next(itertools.islice(iter(self), index, index + 1), _SENTINEL)
        if result is _SENTINEL:
            raise IndexError("index out of range")
        return cast(NDArray[np.uint8], result)

    def __array__(self, dtype: np.typing.DTypeLike | None = None) -> NDArray[np.uint8]:
        """Support numpy array conversion for uniform-length vectors only.

        Warning: This materializes all vectors into memory and requires
        all vectors to have the same length.

        :param dtype: Optional dtype for the result array
        :return: 2D numpy array of all unpadded vectors
        :raises ValueError: If vectors have different lengths
        """
        vectors = list(self)
        if not vectors:
            return np.array([], dtype=dtype or np.uint8)

        # Check if all vectors have the same length
        lengths = {len(v) for v in vectors}
        if len(lengths) > 1:
            raise ValueError(
                f"Cannot convert to array: vectors have different lengths {lengths}. Use list(idx.vectors) instead."
            )

        result = np.vstack(vectors)
        if dtype is not None and result.dtype != dtype:
            return result.astype(dtype)
        return result

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ShardedNphdIndexedVectors(count={len(self)})"


class ShardedNphdIndex(ShardedIndex):
    """Sharded index for variable-length binary bit-vectors with NPHD metric.

    Combines ShardedIndex's automatic sharding with NphdIndex's support for
    variable-length vectors and Normalized Prefix Hamming Distance metric.

    CONCURRENCY: Single-process only. No file locking. Use async/await within
    a single process for concurrent access.
    """

    def __init__(
        self,
        *,
        max_dim: int | None = None,
        path: str | os.PathLike,
        shard_size: int = DEFAULT_SHARD_SIZE,
        connectivity: int | None = None,
        expansion_add: int | None = None,
        expansion_search: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a sharded NPHD index.

        :param max_dim: Maximum bits per vector (auto-detected from existing shards if omitted)
        :param path: Directory path for shard storage (required)
        :param shard_size: Size limit in bytes before rotating shards (default 1GB)
        :param connectivity: HNSW connectivity parameter (M)
        :param expansion_add: Search depth on insertions (efConstruction)
        :param expansion_search: Search depth on queries (ef)
        """
        # Store path early for _resolve_max_dim
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

        # Resolve max_dim from existing shards if not provided
        resolved_max_dim = self._resolve_max_dim(max_dim)
        self._max_dim = resolved_max_dim
        self._max_bytes = resolved_max_dim // 8

        # Remove NPHD-incompatible params (computed from max_dim)
        kwargs.pop("ndim", None)
        kwargs.pop("metric", None)
        kwargs.pop("dtype", None)

        super().__init__(
            ndim=resolved_max_dim + 8,  # +8 bits for length signal byte
            metric=MetricKind.NPHD,
            dtype="b1",  # ScalarKind.B1
            path=path,
            shard_size=shard_size,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
            **kwargs,
        )

    def _resolve_max_dim(self, max_dim: int | None) -> int:
        """Resolve max_dim from existing shards or use provided value.

        :param max_dim: Provided max_dim or None for auto-detection
        :return: Resolved max_dim value
        :raises ValueError: If max_dim is None and no existing shards found
        """
        # Check for existing shards
        existing_shards = sorted(
            self._path.glob("shard_*.usearch"),
            key=lambda p: int(p.stem.split("_")[1]),
        )

        if not existing_shards:
            if max_dim is None:
                raise ValueError("max_dim is required when creating a new index (no existing shards found)")
            return max_dim

        if max_dim is not None:
            return max_dim

        # Read metadata from first shard and compute max_dim
        meta = Index.metadata(str(existing_shards[0]))
        if meta is None:  # pragma: no cover - shard files are always valid in practice
            raise ValueError("max_dim is required (failed to read shard metadata)")

        # ndim = max_dim + 8 (length signal byte), so max_dim = ndim - 8
        return meta["dimensions"] - 8

    def _create_shard(self) -> Index:
        """Create a new Index shard with NPHD metric.

        Uses raw Index (not NphdIndex) so padding is handled at ShardedNphdIndex level.
        """
        return Index(
            ndim=self._max_dim + 8,
            metric=MetricKind.NPHD,
            dtype="b1",
            connectivity=self._config.get("connectivity"),
            expansion_add=self._config.get("expansion_add"),
            expansion_search=self._config.get("expansion_search"),
        )

    def _restore_shard(self, path: Path, view: bool) -> Index | None:
        """Restore an Index shard from disk."""
        meta = Index.metadata(str(path))
        if meta is None:  # pragma: no cover - shard files are always valid in practice
            return None
        shard = Index(
            ndim=meta["dimensions"],
            metric=MetricKind.NPHD,
            dtype=meta["kind_scalar"],
        )
        if view:
            shard.view(str(path))
        else:
            shard.load(str(path))
        return shard

    @property
    def max_dim(self) -> int:
        """Maximum number of bits per vector."""
        return self._max_dim

    @property
    def max_bytes(self) -> int:
        """Maximum number of bytes per vector."""
        return self._max_bytes

    def add(
        self,
        keys: int | None | Any,
        vectors: NDArray[Any] | Sequence[NDArray[Any]],
        *,
        copy: bool = True,
        threads: int = 0,
        log: str | bool = False,
        progress: Callable[[int, int], bool] | None = None,
    ) -> int | NDArray[np.uint64]:
        """Add variable-length binary vectors to the index.

        Pads vectors before adding to ensure consistent storage across shards.

        :param keys: Integer key(s) or None for auto-generation
        :param vectors: Single vector or batch of variable-length vectors to add
        :param copy: Whether to copy vectors into index
        :param threads: Number of threads (0 = auto)
        :param log: Enable progress logging
        :param progress: Progress callback
        :return: Key(s) for added vectors
        """
        # Handle single vector - wrap in list for padding
        vecs: Any = [vectors] if hasattr(vectors, "ndim") and vectors.ndim == 1 else vectors

        # Pad vectors to uniform size
        padded = pad_vectors(vecs, self._max_bytes)

        # Call parent add with padded vectors
        return super().add(keys, padded, copy=copy, threads=threads, log=log, progress=progress)

    def upsert(self, keys: Any, vectors: NDArray[Any] | Sequence[NDArray[Any]], **kwargs: Any) -> int | NDArray:
        """Insert or update variable-length vectors by key.

        Handles ragged/mixed-length vectors that np.asarray cannot stack.

        :param keys: Key(s) — None not accepted
        :param vectors: Single vector or batch of variable-length vectors
        :return: Key(s) for stored vectors
        :raises ValueError: If keys is None or multi=True
        :raises RuntimeError: If index is read-only
        """
        self._check_writable()
        if keys is None:
            raise ValueError("upsert() requires explicit keys")
        if self._config.get("multi"):
            raise ValueError("upsert() requires multi=False")
        if self._is_single_key(keys):
            self.remove(keys)
            return self.add(keys, vectors, **kwargs)

        # Batch path — handle variable-length vectors without np.asarray
        keys_arr = self._normalize_batch_keys(keys)
        vecs: Any = [vectors] if hasattr(vectors, "ndim") and vectors.ndim == 1 else vectors
        if isinstance(vecs, np.ndarray) and vecs.ndim == 2:
            vectors_seq: Any = vecs
        else:
            vectors_seq = list(vecs)
        if len(keys_arr) != len(vectors_seq):
            raise ValueError(f"Number of keys ({len(keys_arr)}) must match number of vectors ({len(vectors_seq)})")
        # Reverse-unique: flip, unique, flip back — keeps last occurrence
        _, unique_idx = np.unique(keys_arr[::-1], return_index=True)
        unique_idx = len(keys_arr) - 1 - unique_idx
        unique_idx.sort()
        if len(unique_idx) < len(keys_arr):
            keys_arr = keys_arr[unique_idx]
            if isinstance(vectors_seq, np.ndarray):
                vectors_seq = cast(Any, vectors_seq)[unique_idx]
            else:
                vectors_seq = [vectors_seq[i] for i in unique_idx]

        self.remove(keys_arr)
        return self.add(keys_arr, vectors_seq, **kwargs)

    def add_once(
        self,
        keys: int | Any,
        vectors: NDArray[Any] | Sequence[NDArray[Any]],
        **kwargs: Any,
    ) -> int | NDArray | None:
        """Add variable-length vectors, skipping keys that already exist.

        First-write-wins: existing keys kept unchanged. Batch duplicates
        deduplicated (first occurrence kept).

        Not atomic under concurrent writes — caller must serialize if needed.

        :param keys: Integer key(s) — None not accepted
        :param vectors: Single vector or batch of variable-length vectors
        :param kwargs: Additional arguments passed to add()
        :return: Key(s) added, empty array if all skipped, None if single key skipped
        :raises ValueError: If keys is None or keys/vectors length mismatch
        """
        if keys is None:
            raise ValueError("add_once() requires explicit keys")
        if self._is_single_key(keys):
            if self.contains(keys):
                return None
            return self.add(keys, vectors, **kwargs)
        # Batch path — vectors may be mixed-length, cannot use np.asarray
        keys_arr = self._normalize_batch_keys(keys)
        # Match add() normalization: treat 1D ndarray as a single vector
        vecs: Any = [vectors] if hasattr(vectors, "ndim") and vectors.ndim == 1 else vectors
        if isinstance(vecs, np.ndarray) and vecs.ndim == 2:
            vectors_seq: Any = vecs
        else:
            vectors_seq = list(vecs)
        if len(keys_arr) != len(vectors_seq):
            raise ValueError(f"Number of keys ({len(keys_arr)}) must match number of vectors ({len(vectors_seq)})")
        # Deduplicate within batch — keep first occurrence
        _, first_indices = np.unique(keys_arr, return_index=True)
        first_indices = np.sort(first_indices)
        keys_arr = keys_arr[first_indices]
        if isinstance(vectors_seq, np.ndarray):
            vectors_seq = cast(Any, vectors_seq)[first_indices]
        else:
            vectors_seq = [vectors_seq[i] for i in first_indices]
        # Filter out keys already in index
        exists = np.asarray(self.contains(keys_arr), dtype=bool)
        new_mask = ~exists
        if not new_mask.any():
            return np.array([], dtype=self._key_dtype)
        new_keys = keys_arr[new_mask]
        new_indices = np.where(new_mask)[0]
        if isinstance(vectors_seq, np.ndarray):
            new_vectors = cast(Any, vectors_seq)[new_indices]
        else:
            new_vectors = [vectors_seq[i] for i in new_indices]
        return self.add(new_keys, new_vectors, **kwargs)

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
        """Search for nearest neighbors of query vector(s).

        Pads query vectors before searching to match stored format.

        :param vectors: Query vector or batch of variable-length vectors to query
        :param count: Maximum number of nearest neighbors to return per query
        :param radius: Maximum distance for results
        :param threads: Number of threads (0 = auto)
        :param exact: Perform exact search
        :param log: Enable progress logging
        :param progress: Progress callback
        :return: Matches for single query, BatchMatches for batch
        """
        # Detect single vector before padding (avoid np.asarray which fails on ragged lists)
        is_single = hasattr(vectors, "ndim") and vectors.ndim == 1

        # Handle single vector - wrap in list for padding
        vecs: Any = [vectors] if is_single else vectors

        # Pad vectors to uniform size (handles variable-length vectors)
        padded = pad_vectors(vecs, self._max_bytes)

        # For single query, pass 1D array so parent returns Matches
        if is_single:
            padded = padded[0]

        # Call parent search with padded vectors
        return super().search(
            padded,
            count=count,
            radius=radius,
            threads=threads,
            exact=exact,
            log=log,
            progress=progress,
        )

    def get(
        self,
        keys: int | Any,
        dtype: Any = None,
    ) -> NDArray[Any] | list | None:
        """Retrieve unpadded variable-length vectors by key(s) from any shard.

        :param keys: Integer key(s) to lookup
        :param dtype: Optional data type for returned vectors
        :return: Unpadded vector(s) or None for missing keys
        """
        # Single key case
        if self._is_single_key(keys):
            result = super().get(keys, dtype=dtype)
            if result is None:
                return None
            return unpad_vectors(np.asarray(result).reshape(1, -1))[0]

        # Multiple keys case - parent returns list with None for missing keys
        keys_seq = cast(Sequence[int], keys)
        results = super().get(keys_seq, dtype=dtype)
        if results is None:  # pragma: no cover - defensive check
            return [None] * len(keys_seq)

        # Unpad found vectors, preserve None for missing keys
        return [unpad_vectors(r.reshape(1, -1))[0] if r is not None else None for r in results]

    def _load_existing(self) -> None:
        """Load existing shards and sync max_dim from loaded shard."""
        super()._load_existing()
        if self._active_shard is not None:
            # Compute max_dim from shard's ndim (ndim = max_dim + 8 for length byte)
            self._max_dim = self._active_shard.ndim - 8
            self._max_bytes = self._max_dim // 8
        else:
            # Read-only mode: derive max_dim from config (set by parent's _load_existing)
            self._max_dim = self._config["ndim"] - 8
            self._max_bytes = self._max_dim // 8

    @property
    def vectors(self) -> ShardedNphdIndexedVectors:
        """Lazy iterator over all unpadded vectors across all shards.

        Returns a ShardedNphdIndexedVectors object that supports:
        - Iteration: for vec in idx.vectors
        - Length: len(idx.vectors)
        - Indexing: idx.vectors[0], idx.vectors[-1]
        - Slicing: idx.vectors[:10]
        - Numpy conversion: np.asarray(idx.vectors) (requires uniform vector lengths)

        Vectors are returned unpadded (variable-length), consistent with the get() API.
        This is a live view - reflects current state at iteration time.

        :return: ShardedNphdIndexedVectors iterator
        """
        return ShardedNphdIndexedVectors(self)

    def __repr__(self) -> str:
        """Return string representation of the sharded NPHD index."""
        ro = ", read_only" if self._read_only else ""
        return f"ShardedNphdIndex({self.size} vectors in {self.shard_count} shards, max_dim={self._max_dim}, path={self._path}{ro})"


class ShardedNphdIndex128(_UuidKeyMixin, ShardedNphdIndex):
    """Sharded NPHD index with 128-bit UUID keys.

    Combines ShardedNphdIndex's variable-length vector support with 128-bit
    composite keys represented as bytes(16) for single keys and np.dtype('V16')
    arrays for batches.

    Auto-generation of keys (keys=None) is not supported — all keys must be
    provided explicitly.
    """

    def __init__(self, *, path: str | os.PathLike, **kwargs: Any) -> None:
        """Initialize a 128-bit sharded NPHD index.

        :param path: Directory path for shard storage
        :param kwargs: Passed to ShardedNphdIndex (key_kind is absorbed if present)
        """
        kwargs.pop("key_kind", None)
        super().__init__(path=path, **kwargs)

    def _create_shard(self) -> Index:
        """Create a uuid-keyed NPHD shard."""
        return Index(
            ndim=self._max_dim + 8,
            metric=MetricKind.NPHD,
            dtype="b1",
            connectivity=self._config.get("connectivity"),
            expansion_add=self._config.get("expansion_add"),
            expansion_search=self._config.get("expansion_search"),
            key_kind="uuid",
        )

    def _restore_shard(self, path: Path, view: bool) -> Index | None:
        """Restore a uuid-keyed NPHD shard from disk."""
        meta = Index.metadata(str(path))
        if meta is None:  # pragma: no cover
            return None
        shard = Index(
            ndim=meta["dimensions"],
            metric=MetricKind.NPHD,
            dtype=meta["kind_scalar"],
            key_kind="uuid",
        )
        if view:
            shard.view(str(path))
        else:
            shard.load(str(path))
        return shard

    def __repr__(self) -> str:
        """Return string representation of the sharded NPHD 128-bit index."""
        ro = ", read_only" if self._read_only else ""
        return f"ShardedNphdIndex128({self.size} vectors in {self.shard_count} shards, max_dim={self._max_dim}, path={self._path}{ro})"
