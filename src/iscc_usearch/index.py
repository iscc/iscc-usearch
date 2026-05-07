"""Drop-in replacement for usearch.index.Index with upsert support and durable saves."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from usearch.index import Index as _Index, ScalarKind

from iscc_usearch.utils import durable_write

__all__ = ["Index"]

# NumPy dtype for 128-bit UUID keys
_UUID_DTYPE = np.dtype("V16")


class Index(_Index):
    """Index with upsert support and durable file saves."""

    def save(self, path_or_buffer=None, progress=None, release_gil=False):
        """Save index to file or buffer.

        When saving to a file, serializes to an in-memory buffer first, then writes
        durably to disk for minimal IOPS and power-loss safety.

        :param path_or_buffer: File path to save to, or None to return serialized bytes.
        :param progress: Optional progress callback.
        :param release_gil: Release the GIL during serialization (requires exclusive access).
        :return: Serialized buffer when saving to memory, None when saving to file.
        """
        if path_or_buffer is not None:
            data = self._compiled.save_index_to_buffer(progress, release_gil)
            durable_write(data, Path(os.fspath(path_or_buffer)))
            return None
        return super().save(progress=progress, release_gil=release_gil)

    def upsert(
        self,
        keys: int | bytes | Sequence[int] | NDArray,
        vectors: NDArray,
        **kwargs: Any,
    ) -> NDArray:
        """
        Insert or update vectors by key (true idempotent semantics).

        Calling upsert(key, vec) always results in vec being stored for that key.
        For existing keys, updates only if the vector differs (conditional update).

        Supports both uint64 and 128-bit UUID keys.

        :param keys: Key(s) — int/Sequence[int] for uint64, bytes(16)/V16 ndarray for UUID.
        :param vectors: Vector(s) to upsert (must be uniform length in batch).
        :param kwargs: Additional arguments passed to add().
        :return: Array of keys (uint64 or V16 dtype).
        :raises ValueError: If keys is None or keys/vectors length mismatch.
        """
        if keys is None:
            raise ValueError("upsert() requires explicit keys. Auto-generation (keys=None) is not supported.")

        is_uuid = getattr(self, "_key_kind", None) == ScalarKind.UUID

        # Normalize inputs
        if is_uuid:
            if isinstance(keys, bytes):
                if len(keys) != 16:
                    raise ValueError(f"UUID key must be exactly 16 bytes, got {len(keys)}")
                keys_arr = np.array([keys], dtype=_UUID_DTYPE)
            else:
                keys_arr = np.atleast_1d(np.asarray(keys))
        else:
            keys_arr = np.atleast_1d(np.asarray(keys, dtype=np.uint64))
        vectors_arr = np.atleast_2d(np.asarray(vectors))

        if len(keys_arr) != len(vectors_arr):
            raise ValueError(f"Number of keys ({len(keys_arr)}) must match number of vectors ({len(vectors_arr)})")

        if len(keys_arr) == 0:
            return np.array([], dtype=keys_arr.dtype)

        # Handle internal duplicates: keep LAST occurrence (upsert = last write wins)
        reversed_keys = keys_arr[::-1]
        _, unique_rev_idx = np.unique(reversed_keys, return_index=True)
        last_indices = np.sort(len(keys_arr) - 1 - unique_rev_idx)

        unique_keys = keys_arr[last_indices]
        unique_vectors = vectors_arr[last_indices]

        # Check which keys exist
        exists_mask = np.asarray(self.contains(unique_keys), dtype=bool)

        # Add new keys
        new_mask = ~exists_mask
        if new_mask.any():
            self.add(unique_keys[new_mask], unique_vectors[new_mask], **kwargs)

        # Update existing keys only if vector differs
        if exists_mask.any():
            existing_keys = unique_keys[exists_mask]
            existing_new_vectors = unique_vectors[exists_mask]
            current_vectors = np.asarray(self.get(existing_keys))

            differs = ~np.all(current_vectors == existing_new_vectors, axis=1)
            if differs.any():
                keys_to_update = existing_keys[differs]
                vectors_to_update = existing_new_vectors[differs]
                self.remove(keys_to_update)
                self.add(keys_to_update, vectors_to_update, **kwargs)

        return keys_arr.copy()
