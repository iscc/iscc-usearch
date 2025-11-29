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

__all__ = ["ShardedNphdIndex"]


class ShardedNphdIndex(ShardedIndex):
    """Sharded index for variable-length binary bit-vectors with NPHD metric.

    Combines ShardedIndex's automatic sharding with NphdIndex's support for
    variable-length vectors and Normalized Prefix Hamming Distance metric.

    CONCURRENCY: Single-process only. No file locking. Use async/await within
    a single process for concurrent access.

    :param max_dim: Maximum number of bits per vector (default 256)
    :param path: Directory path for shard storage (required)
    :param shard_size: Size limit in bytes before rotating shards (default 1GB)
    :param view: Load existing shards in view mode only (read-only)
    :param connectivity: HNSW connectivity parameter (M)
    :param expansion_add: Search depth on insertions (efConstruction)
    :param expansion_search: Search depth on queries (ef)
    """

    def __init__(
        self,
        *,
        max_dim: int = 256,
        path: str | os.PathLike,
        **kwargs: Any,
    ) -> None:
        """Initialize a sharded NPHD index."""
        self._max_dim = max_dim
        self._max_bytes = max_dim // 8

        # Remove NPHD-incompatible params (computed from max_dim)
        kwargs.pop("ndim", None)
        kwargs.pop("metric", None)
        kwargs.pop("dtype", None)

        super().__init__(
            ndim=max_dim + 8,  # +8 bits for length signal byte
            metric=create_nphd_metric(),
            dtype="b1",  # ScalarKind.B1
            path=path,
            **kwargs,
        )

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

        When view=True, disables key lookups to skip expensive hash map population
        (~2x speedup). Safe because view shards only support search(), not get/contains.
        """
        meta = Index.metadata(str(path))
        if meta is None:  # pragma: no cover - shard files are always valid in practice
            return None
        shard = Index(
            ndim=meta["dimensions"],
            metric=create_nphd_metric(),
            dtype=meta["kind_scalar"],
            enable_key_lookups=not view,  # Disable for view shards (2x speedup)
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
        """Retrieve unpadded variable-length vectors by key(s).

        Note: View shards do not support get(). Only keys in the active shard
        can be retrieved.

        :param keys: Integer key(s) to lookup
        :param dtype: Optional data type for returned vectors
        :return: Unpadded vector(s) or None for missing keys
        """
        # Handle single key - check existence first (workaround for usearch bug)
        if isinstance(keys, int):
            if not self.contains(keys):
                return None
            results = super().get(keys, dtype=dtype)
            if results is None:  # pragma: no cover - defensive check
                return None
            return unpad_vectors(results.reshape(1, -1))[0]

        # Handle multiple keys - check existence for each
        exists = self.contains(keys)
        results = super().get(keys, dtype=dtype)

        if results is None:  # pragma: no cover - defensive check
            return [None] * len(keys)

        # Unpad existing results, return None for missing keys
        unpacked = []
        results_list = list(results)  # Explicit list conversion for type safety
        for r, e in zip(results_list, exists):
            if not e:
                unpacked.append(None)
            elif r is None:  # pragma: no cover - defensive check
                unpacked.append(None)
            else:
                unpacked.append(unpad_vectors(r.reshape(1, -1))[0])
        return unpacked

    def load(
        self,
        path_or_buffer: str | os.PathLike | None = None,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None:
        """Load shards from directory and sync max_dim from loaded shard.

        :param path_or_buffer: Ignored (uses internal path management)
        :param progress: Progress callback
        """
        super().load(path_or_buffer, progress)
        # ShardedIndex.load() always creates active_shard if none exists
        if self._active_shard is not None:  # pragma: no branch
            # Compute max_dim from shard's ndim (ndim = max_dim + 8 for length byte)
            self._max_dim = self._active_shard.ndim - 8
            self._max_bytes = self._max_dim // 8

    def view(
        self,
        path_or_buffer: str | os.PathLike | None = None,
        progress: Callable[[int, int], bool] | None = None,
    ) -> None:
        """View shards from directory (read-only) and sync max_dim.

        :param path_or_buffer: Ignored (uses internal path management)
        :param progress: Progress callback
        """
        super().view(path_or_buffer, progress)
        # max_dim is already set from restore() or constructor

    @staticmethod
    def restore(
        path: str | os.PathLike,
        view: bool = False,
        **kwargs: Any,
    ) -> ShardedNphdIndex | None:
        """Restore a ShardedNphdIndex from a directory.

        :param path: Directory containing shard files
        :param view: Open in view mode (read-only)
        :param kwargs: Additional arguments passed to constructor
        :return: Restored ShardedNphdIndex or None if invalid
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

        meta = Index.metadata(str(shard_files[0]))
        if not meta:  # pragma: no cover - Index.metadata raises on invalid files
            return None

        max_dim = meta["dimensions"] - 8  # Subtract length signal byte
        return ShardedNphdIndex(max_dim=max_dim, path=path, view=view, **kwargs)

    def __repr__(self) -> str:
        """Return string representation of the sharded NPHD index."""
        return f"ShardedNphdIndex({self.size} vectors in {self.shard_count} shards, max_dim={self._max_dim}, path={self._path})"
