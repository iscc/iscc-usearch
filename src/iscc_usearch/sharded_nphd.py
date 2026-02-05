"""Sharded NPHD index for scalable variable-length binary bit-vector storage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from usearch.index import BatchMatches, Index, Matches

from iscc_usearch.metrics import create_nphd_metric
from iscc_usearch.nphd import pad_vectors, unpad_vectors
from iscc_usearch.sharded import ShardedIndex

__all__ = ["ShardedNphdIndex", "ShardedNphdIndexedVectors"]


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

    def __iter__(self):
        """Yield unpadded vectors from all shards lazily.

        Iterates through view shards first, then active shard.
        """
        for idx in self._index._viewed_indexes:
            for vec in idx.vectors:
                yield unpad_vectors(vec.reshape(1, -1))[0]
        if self._index._active_shard is not None:
            for vec in self._index._active_shard.vectors:
                yield unpad_vectors(vec.reshape(1, -1))[0]

    def __getitem__(self, index: int | slice) -> NDArray[np.uint8] | list[NDArray[np.uint8]]:
        """Support indexing and slicing.

        :param index: Integer index or slice
        :return: Single unpadded vector or list of unpadded vectors
        """
        if isinstance(index, slice):
            # Handle slicing by materializing the requested range
            vectors = list(self)
            return vectors[index]
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
                    vec = idx.vectors[index - current]
                    return unpad_vectors(vec.reshape(1, -1))[0]
                current += shard_len

            if self._index._active_shard is not None:
                vec = self._index._active_shard.vectors[index - current]
                return unpad_vectors(vec.reshape(1, -1))[0]

            raise IndexError("index out of range")  # pragma: no cover

    def __array__(self, dtype: np.typing.DTypeLike = None) -> NDArray[np.uint8]:
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

    :param max_dim: Maximum bits per vector (auto-detected from existing shards if omitted)
    :param path: Directory path for shard storage (required)
    :param shard_size: Size limit in bytes before rotating shards (default 1GB)
    :param connectivity: HNSW connectivity parameter (M)
    :param expansion_add: Search depth on insertions (efConstruction)
    :param expansion_search: Search depth on queries (ef)
    """

    def __init__(
        self,
        *,
        max_dim: int | None = None,
        path: str | os.PathLike,
        **kwargs: Any,
    ) -> None:
        """Initialize a sharded NPHD index."""
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
            metric=create_nphd_metric(),
            dtype="b1",  # ScalarKind.B1
            path=path,
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
            metric=create_nphd_metric(),
            dtype="b1",
            connectivity=self._config.get("connectivity"),
            expansion_add=self._config.get("expansion_add"),
            expansion_search=self._config.get("expansion_search"),
        )

    def _restore_shard(self, path: Path, view: bool) -> Index | None:
        """Restore an Index shard from disk and restore NPHD metric.

        When enable_key_lookups=False, skips expensive hash map population (~2x speedup)
        but disables contains/get/count operations.
        """
        meta = Index.metadata(str(path))
        if meta is None:  # pragma: no cover - shard files are always valid in practice
            return None
        enable_lookups = self._config.get("enable_key_lookups", True)
        shard = Index(
            ndim=meta["dimensions"],
            metric=create_nphd_metric(),
            dtype=meta["kind_scalar"],
            enable_key_lookups=enable_lookups,
        )
        if view:
            shard.view(str(path))
        else:
            shard.load(str(path))
        # Restore custom NPHD metric (usearch load/view replaces it with standard Hamming)
        metric = create_nphd_metric()
        shard._compiled.change_metric(metric.kind, metric.signature, metric.pointer)
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
        vectors: NDArray[Any],
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
        if hasattr(vectors, "ndim") and vectors.ndim == 1:
            vectors = [vectors]

        # Pad vectors to uniform size
        padded = pad_vectors(vectors, self._max_bytes)

        # Call parent add with padded vectors
        return super().add(keys, padded, copy=copy, threads=threads, log=log, progress=progress)

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
        # Track if original input was single vector
        vectors = np.asarray(vectors)
        is_single = vectors.ndim == 1

        # Handle single vector - wrap in list for padding
        if is_single:
            vectors = [vectors]

        # Pad vectors to uniform size
        padded = pad_vectors(vectors, self._max_bytes)

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

        When enable_key_lookups=True (default), searches all shards.
        When enable_key_lookups=False, returns None for all keys.

        :param keys: Integer key(s) to lookup
        :param dtype: Optional data type for returned vectors
        :return: Unpadded vector(s) or None for missing keys
        """
        # Single key case
        if isinstance(keys, int):
            result = super().get(keys, dtype=dtype)
            if result is None:
                return None
            return unpad_vectors(result.reshape(1, -1))[0]

        # Multiple keys case - parent returns list with None for missing keys
        results = super().get(keys, dtype=dtype)
        if results is None:  # pragma: no cover - defensive check
            return [None] * len(keys)

        # Unpad found vectors, preserve None for missing keys
        return [unpad_vectors(r.reshape(1, -1))[0] if r is not None else None for r in results]

    def _load_existing(self) -> None:
        """Load existing shards and sync max_dim from loaded shard."""
        super()._load_existing()
        # Parent always creates active_shard if none exists
        if self._active_shard is not None:  # pragma: no branch
            # Compute max_dim from shard's ndim (ndim = max_dim + 8 for length byte)
            self._max_dim = self._active_shard.ndim - 8
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
        return f"ShardedNphdIndex({self.size} vectors in {self.shard_count} shards, max_dim={self._max_dim}, path={self._path})"
