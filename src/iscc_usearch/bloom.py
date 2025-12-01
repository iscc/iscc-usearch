"""Scalable bloom filter for efficient key existence checks.

Wraps fastbloom_rs to provide automatic growth as elements are added.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Sequence

from fastbloom_rs import BloomFilter

__all__ = ["ScalableBloomFilter"]

# File format magic bytes for validation
MAGIC = b"ISBF"  # ISCC Scalable Bloom Filter
VERSION = 1


class ScalableBloomFilter:
    """Scalable bloom filter that grows automatically as elements are added.

    Chains multiple fixed-size bloom filters to support unlimited growth while
    maintaining the target false positive rate. Each new filter has progressively
    tighter FPR to keep the overall rate bounded.

    :param initial_capacity: Initial number of elements before first growth
    :param fpr: Target false positive rate (0.0-1.0)
    :param growth_factor: Capacity multiplier for each new filter
    """

    def __init__(
        self,
        initial_capacity: int = 10_000_000,
        fpr: float = 0.01,
        growth_factor: float = 2.0,
    ) -> None:
        """Initialize a scalable bloom filter."""
        self._initial_capacity = initial_capacity
        self._fpr = fpr
        self._growth_factor = growth_factor
        self._filters: list[BloomFilter] = []
        self._capacities: list[int] = []  # Track capacity of each filter
        self._count = 0
        self._add_filter()

    def _add_filter(self) -> None:
        """Add a new bloom filter to the chain."""
        # Calculate capacity for this filter
        if not self._filters:
            capacity = self._initial_capacity
        else:
            capacity = int(self._capacities[-1] * self._growth_factor)

        # Each new filter has tighter FPR to maintain overall target
        # Using geometric series: sum of FPR * r^i converges to FPR / (1-r)
        # With r=0.5, total FPR is bounded at 2x individual FPR
        scale = 0.5 ** len(self._filters)
        filter_fpr = self._fpr * scale

        self._filters.append(BloomFilter(capacity, filter_fpr))
        self._capacities.append(capacity)

    @property
    def count(self) -> int:
        """Approximate number of elements added."""
        return self._count

    @property
    def current_capacity(self) -> int:
        """Total capacity across all filters."""
        return sum(self._capacities)

    @property
    def filter_count(self) -> int:
        """Number of bloom filters in the chain."""
        return len(self._filters)

    def add(self, key: int) -> None:
        """Add a single key to the bloom filter.

        :param key: Integer key to add (uint64)
        """
        # Check if we need to grow (when current filter is "full")
        current_filter_count = self._count - sum(self._capacities[:-1])
        if current_filter_count >= self._capacities[-1]:
            self._add_filter()

        self._filters[-1].add_int(key)
        self._count += 1

    def add_batch(self, keys: Sequence[int]) -> None:
        """Add multiple keys to the bloom filter efficiently.

        Uses native batch operations and handles capacity growth properly.

        :param keys: Sequence of integer keys to add
        """
        if not keys:
            return

        keys_list = list(keys) if not isinstance(keys, list) else keys
        remaining = keys_list

        while remaining:
            # Calculate space left in current filter
            filled_in_previous = sum(self._capacities[:-1])
            current_filter_filled = self._count - filled_in_previous
            space_left = self._capacities[-1] - current_filter_filled

            if space_left <= 0:
                # Current filter is full, add a new one
                self._add_filter()
                continue

            # Take as many keys as fit in current filter
            batch = remaining[:space_left]
            remaining = remaining[space_left:]

            # Add batch to current filter using native operation
            self._filters[-1].add_int_batch(batch)
            self._count += len(batch)

    def contains(self, key: int) -> bool:
        """Check if a key might be in the filter.

        :param key: Integer key to check
        :return: False if definitely not present, True if possibly present
        """
        # Check all filters - return True if any says "maybe"
        # Check newest first (most likely location for recent keys)
        for f in reversed(self._filters):
            if f.contains_int(key):
                return True
        return False

    def contains_batch(self, keys: Sequence[int]) -> list[bool]:
        """Check if multiple keys might be in the filter.

        :param keys: Sequence of integer keys to check
        :return: List of booleans (False=definitely not, True=possibly present)
        """
        return [self.contains(key) for key in keys]

    def clear(self) -> None:
        """Clear all filters and reset to initial state."""
        self._filters.clear()
        self._capacities.clear()
        self._count = 0
        self._add_filter()

    def save(self, path: str | Path) -> None:
        """Save bloom filter to disk.

        File format:
        - 4 bytes: magic ("ISBF")
        - 1 byte: version
        - 8 bytes: count (uint64)
        - 4 bytes: initial_capacity (uint32)
        - 8 bytes: fpr (float64)
        - 8 bytes: growth_factor (float64)
        - 4 bytes: num_filters (uint32)
        - For each filter:
          - 4 bytes: capacity (uint32)
          - 4 bytes: hashes (uint32)
          - 4 bytes: data_len (uint32)
          - data_len bytes: filter data

        :param path: File path to save to
        """
        path = Path(path)
        with open(path, "wb") as f:
            # Header
            f.write(MAGIC)
            f.write(struct.pack("<B", VERSION))
            f.write(struct.pack("<Q", self._count))
            f.write(struct.pack("<I", self._initial_capacity))
            f.write(struct.pack("<d", self._fpr))
            f.write(struct.pack("<d", self._growth_factor))
            f.write(struct.pack("<I", len(self._filters)))

            # Each filter
            for i, bf in enumerate(self._filters):
                data = bytes(bf.get_bytes())
                f.write(struct.pack("<I", self._capacities[i]))
                f.write(struct.pack("<I", bf.hashes()))
                f.write(struct.pack("<I", len(data)))
                f.write(data)

    @classmethod
    def load(cls, path: str | Path) -> ScalableBloomFilter:
        """Load bloom filter from disk.

        :param path: File path to load from
        :return: Restored ScalableBloomFilter
        :raises ValueError: If file format is invalid
        """
        path = Path(path)
        with open(path, "rb") as f:
            # Header
            magic = f.read(4)
            if magic != MAGIC:
                raise ValueError(f"Invalid bloom filter file: bad magic {magic!r}")

            version = struct.unpack("<B", f.read(1))[0]
            if version != VERSION:
                raise ValueError(f"Unsupported bloom filter version: {version}")

            count = struct.unpack("<Q", f.read(8))[0]
            initial_capacity = struct.unpack("<I", f.read(4))[0]
            fpr = struct.unpack("<d", f.read(8))[0]
            growth_factor = struct.unpack("<d", f.read(8))[0]
            num_filters = struct.unpack("<I", f.read(4))[0]

            # Create instance without auto-creating first filter
            instance = cls.__new__(cls)
            instance._initial_capacity = initial_capacity
            instance._fpr = fpr
            instance._growth_factor = growth_factor
            instance._filters = []
            instance._capacities = []
            instance._count = count

            # Load each filter
            for _ in range(num_filters):
                capacity = struct.unpack("<I", f.read(4))[0]
                hashes = struct.unpack("<I", f.read(4))[0]
                data_len = struct.unpack("<I", f.read(4))[0]
                data = f.read(data_len)

                bf = BloomFilter.from_bytes(data, hashes)
                instance._filters.append(bf)
                instance._capacities.append(capacity)

        return instance

    def __len__(self) -> int:
        """Return approximate number of elements."""
        return self._count

    def __contains__(self, key: int) -> bool:
        """Support 'in' operator."""
        return self.contains(key)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ScalableBloomFilter(count={self._count:,}, "
            f"filters={len(self._filters)}, "
            f"capacity={self.current_capacity:,}, "
            f"fpr={self._fpr})"
        )
