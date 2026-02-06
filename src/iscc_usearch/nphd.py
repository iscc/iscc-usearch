"""
Scalable ANNS search for variable-length binary bit-vectors with NPHD metric.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union  # noqa: F401 - used in type comments

import numpy as np
from numba import njit
from numpy.typing import NDArray
from usearch.index import BatchMatches, Matches, ScalarKind  # noqa: F401 - used in type comments

from iscc_usearch.index import Index

from iscc_usearch.metrics import create_nphd_metric

# Type aliases for cleaner type comments
Key = int | None
Keys = Sequence[int] | None
Vector = NDArray[np.uint8]
Vectors = Sequence[NDArray[np.uint8]] | NDArray[np.uint8]


@njit(cache=True)
def pad_vectors(vectors, nbytes):  # pragma: no cover
    # type: (Vectors, int) -> NDArray[np.uint8]
    """
    Add length prefix and padding to a batch of variable-length bit-vectors.

    Prepends each vector with a length byte and pads to uniform size for fixed-size index storage.
    First byte stores original length, followed by vector bytes and zero padding up to nbytes.

    :param vectors: Sequence of variable-length uint8 arrays or 2D uint8 array with uniform-length vectors.
    :param nbytes: Maximum number of bytes per vector (excluding length prefix byte).
    :return: 2D array of shape (batch_size, nbytes + 1) with length-prefixed padded vectors.
    """
    batch_size = len(vectors)
    padded = np.zeros((batch_size, nbytes + 1), dtype=np.uint8)
    for i in range(batch_size):
        vec = vectors[i]
        length = len(vec)
        padded[i, 0] = length
        for j in range(min(length, nbytes)):
            padded[i, j + 1] = vec[j]
    return padded


@njit(cache=True)
def unpad_vectors(padded):  # pragma: no cover
    # type: (NDArray[np.uint8]) -> list[Vector]
    """
    Extract variable-length bit-vectors from length-prefixed padded matrix.

    Reverses the pad_vectors operation by reading length prefix from first byte
    and extracting only the valid data bytes for each vector.

    :param padded: 2D array of shape (batch_size, nbytes + 1) with length-prefixed padded vectors.
    :return: List of variable-length uint8 arrays without padding.
    """
    batch_size = len(padded)
    result = []
    for i in range(batch_size):
        length = int(padded[i, 0])
        vector = padded[i, 1 : length + 1].copy()
        result.append(vector)
    return result


class NphdIndex(Index):
    """Fast approximate nearest neighbor search for variable-length binary bit-vectors.

    Supports Normalized Prefix Hamming Distance (NPHD) metric and packed binary vectors
    as np.uint8 arrays of variable length. Vector keys must be integers.

    CONCURRENCY: Single-process only. The underlying .usearch files have no file locking
    or multi-process coordination. Running multiple processes against the same index may
    corrupt data. Use a single process with async/await for concurrent connections.

    UPSERT: Batch upsert requires uniform-length vectors. For variable-length batch upsert,
    call upsert() individually for each vector: `for k, v in zip(keys, vecs): idx.upsert(k, v)`
    """

    def __init__(self, max_dim=256, **kwargs):
        # type: (int, Any) -> None
        """Create a new NPHD index."""
        assert max_dim <= 256, f"`max_dim` must be <= 256 (got {max_dim}); the NPHD metric supports at most 256 bits"
        assert max_dim % 8 == 0, f"`max_dim` must be a multiple of 8 (got {max_dim})"

        self.max_dim = max_dim
        self.max_bytes = max_dim // 8

        assert "ndim" not in kwargs, "`ndim` is calculated from `max_dim`"
        assert "metric" not in kwargs, "`metric` is set automatically (NPHD)"
        assert "dtype" not in kwargs, "`dtype` is set automatically (ScalarKind.B1)"

        metric = create_nphd_metric()

        super().__init__(
            ndim=max_dim + 8,  # + 8 bits for length signal byte
            metric=metric,
            dtype=ScalarKind.B1,
            **kwargs,
        )

    def add(self, keys, vectors, **kwargs):
        # type: (Key | Sequence[int], Vectors, Any) -> NDArray[np.uint64]
        """
        Add variable-length binary vectors to the index.

        :param keys: Integer key(s) or None for auto-generation
        :param vectors: Single vector, 2D array of uniform vectors, or list of variable-length vectors
        :param kwargs: Additional arguments passed to parent Index.add()
        :return: Array of keys for added vectors
        """
        # Handle single vector - wrap in list for padding
        if hasattr(vectors, "ndim") and vectors.ndim == 1:
            vectors = [vectors]

        # Pad vectors to uniform size
        padded = pad_vectors(vectors, self.max_bytes)

        # Call parent add with padded vectors
        return super().add(keys, padded, **kwargs)

    def get(self, keys, dtype=None):
        # type: (int | Sequence[int], Any) -> Vector | list | None
        """
        Retrieve unpadded variable-length vectors by key(s).

        :param keys: Integer key(s) to lookup
        :param dtype: Optional data type (defaults to index dtype)
        :return: Unpadded vector(s) or None for missing keys
        """
        results = super().get(keys, dtype=dtype)

        if results is None:
            return None

        if isinstance(results, np.ndarray):
            if results.ndim == 1:
                return unpad_vectors(results.reshape(1, -1))[0]
            return unpad_vectors(results)

        return [
            None if r is None else unpad_vectors(r.reshape(1, -1))[0] if r.ndim == 1 else unpad_vectors(r)
            for r in results
        ]

    def search(self, vectors, count=10, **kwargs):
        # type: (Vectors, int, Any) -> Union[Matches, BatchMatches]
        """
        Search for nearest neighbors of query vector(s).

        :param vectors: Single vector or batch of variable-length vectors to query
        :param count: Maximum number of nearest neighbors to return per query
        :param kwargs: Additional arguments passed to parent Index.search()
        :return: Matches for single query or BatchMatches for batch queries
        :raises ValueError: If count < 1
        """

        # Handle single vector - wrap in list for padding
        if hasattr(vectors, "ndim") and vectors.ndim == 1:
            vectors = [vectors]

        # Pad vectors to uniform size
        padded = pad_vectors(vectors, self.max_bytes)

        # Call parent search with padded vectors
        return super().search(padded, count=count, **kwargs)

    def load(self, path_or_buffer=None, progress=None):
        # type: (Any, Any) -> None
        """
        Load index from file or buffer and restore max_dim from saved ndim.

        CRITICAL: After loading, we must restore the custom NPHD metric because
        usearch's load() overwrites it with the saved metric (standard Hamming).

        :param path_or_buffer: Path or buffer to load from (defaults to self.path)
        :param progress: Optional progress callback
        """
        super().load(path_or_buffer, progress)
        self.max_dim = self.ndim - 8
        self.max_bytes = self.max_dim // 8

        # Restore custom NPHD metric (usearch load() replaces it with standard Hamming)
        metric = create_nphd_metric()
        self._compiled.change_metric(metric.kind, metric.signature, metric.pointer)

    def view(self, path_or_buffer=None, progress=None):
        # type: (Any, Any) -> None
        """
        Memory-map index from file or buffer and restore max_dim from saved ndim.

        CRITICAL: After viewing, we must restore the custom NPHD metric because
        usearch's view() overwrites it with the saved metric (standard Hamming).

        :param path_or_buffer: Path or buffer to view from (defaults to self.path)
        :param progress: Optional progress callback
        """
        super().view(path_or_buffer, progress)
        self.max_dim = self.ndim - 8
        self.max_bytes = self.max_dim // 8

        # Restore custom NPHD metric (usearch view() replaces it with standard Hamming)
        metric = create_nphd_metric()
        self._compiled.change_metric(metric.kind, metric.signature, metric.pointer)

    def copy(self):
        # type: () -> NphdIndex
        """
        Create a copy of this index.

        :return: New NphdIndex with same configuration and data
        """
        result = NphdIndex(
            max_dim=self.max_dim,
            connectivity=self.connectivity,
            expansion_add=self.expansion_add,
            expansion_search=self.expansion_search,
        )
        result._compiled = self._compiled.copy()
        return result

    @staticmethod
    def restore(path_or_buffer, view=False, **kwargs):
        # type: (Any, bool, Any) -> NphdIndex | None
        """
        Restore a NphdIndex from a saved file or buffer.

        :param path_or_buffer: Path or buffer to restore from
        :param view: If True, memory-map the index instead of loading
        :param kwargs: Additional arguments passed to NphdIndex constructor
        :return: Restored NphdIndex or None if file is invalid
        """
        meta = Index.metadata(path_or_buffer)
        if not meta:
            return None

        max_dim = meta["dimensions"] - 8
        index = NphdIndex(max_dim=max_dim, **kwargs)

        if view:
            index.view(path_or_buffer)
        else:
            index.load(path_or_buffer)

        return index
