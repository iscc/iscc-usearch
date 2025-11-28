"""
Drop-in replacement for usearch.index.Index with bug fixes.

Workarounds for:
- https://github.com/unum-cloud/USearch/issues/494
  get() returns garbage data for non-existent keys instead of None.
- search() with count=0 causes segmentation fault.

TODO: Remove this module when usearch releases fixes, then import Index
directly from usearch.index in nphd.py.
"""

from usearch.index import Index as _Index

__all__ = ["Index"]


class Index(_Index):
    """Index with bug fixes for get() and search() methods."""

    def get(self, keys, dtype=None):
        """Retrieve vectors by key(s), returning None for non-existent keys."""
        # Handle single key
        if isinstance(keys, int):
            if not self.contains(keys):
                return None
            return super().get(keys, dtype=dtype)

        # Handle multiple keys - check existence and filter results
        exists = self.contains(keys)
        results = super().get(keys, dtype=dtype)

        # Replace garbage data with None for non-existent keys
        return [r if e else None for r, e in zip(results, exists)]

    def search(self, vectors, count=10, **kwargs):
        """Search for nearest neighbors, with guard against count < 1 segfault."""
        if count < 1:
            raise ValueError("count must be >= 1")
        return super().search(vectors, count=count, **kwargs)
