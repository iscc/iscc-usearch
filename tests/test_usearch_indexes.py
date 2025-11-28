"""
Tests for usearch Indexes class behavior.

These tests verify the behavior of the Indexes class which manages multiple
Index instances for distributed search. Issues discovered here should be
reported upstream to unum-cloud/usearch.

BUG REPORT: Indexes class segfaults when initialized with multiple paths.
- Tested on: Windows 10, Python 3.12, usearch 2.21.0
- Single path works fine
- Two or more paths cause access violation in merge_paths
- GitHub issue: https://github.com/unum-cloud/usearch/issues/643

WORKAROUND: Instead of Indexes(paths=[...], view=True), use:
    1. Load each index separately with Index.restore(path, view=True)
    2. Create empty Indexes()
    3. Merge each index with combined._compiled.merge(idx._compiled)

Minimal reproduction:
    from usearch.index import Index, Indexes, MetricKind, ScalarKind
    import numpy as np
    import tempfile
    from pathlib import Path

    tmpdir = Path(tempfile.mkdtemp())
    paths = []
    for i in range(2):
        idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
        idx.add(i, np.array([i, 2, 3, 4], dtype=np.uint8))
        path = tmpdir / f"shard_{i}.usearch"
        idx.save(str(path))
        paths.append(str(path))
        del idx

    # This crashes with segfault:
    indexes = Indexes(paths=paths, view=True)
"""

import numpy as np
import pytest
from usearch.index import Index, Indexes, MetricKind, ScalarKind


@pytest.fixture
def two_saved_indexes(tmp_path):
    """Create and save two separate index files."""
    paths = []
    for i in range(2):
        idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
        # Add distinct vectors to each shard
        idx.add(i * 10, np.array([i, 2, 3, 4], dtype=np.uint8))
        idx.add(i * 10 + 1, np.array([i, 5, 6, 7], dtype=np.uint8))
        path = tmp_path / f"shard_{i}.usearch"
        idx.save(str(path))
        paths.append(str(path))
        del idx
    return paths


class TestIndexesCreation:
    """Tests for Indexes class instantiation."""

    def test_indexes_from_empty_paths_list(self):
        """Indexes can be created with empty paths list."""
        indexes = Indexes(paths=[])
        assert len(indexes) == 0

    def test_indexes_from_single_path(self, two_saved_indexes):
        """Indexes can be created from a single index file."""
        indexes = Indexes(paths=[two_saved_indexes[0]], view=True)
        assert len(indexes) == 2  # Two vectors in first shard

    @pytest.mark.skip(reason="Segfaults with usearch 2.21.0 - see module docstring")
    def test_indexes_from_multiple_paths(self, two_saved_indexes):
        """Indexes can be created from multiple index files."""
        indexes = Indexes(paths=two_saved_indexes, view=True)
        assert len(indexes) == 4  # Two vectors per shard, two shards

    @pytest.mark.skip(reason="Segfaults with usearch 2.21.0 - see module docstring")
    def test_indexes_view_mode(self, two_saved_indexes):
        """Indexes in view mode are memory-mapped."""
        indexes = Indexes(paths=two_saved_indexes, view=True)
        # Should be able to search without loading full data
        assert len(indexes) == 4


class TestIndexesSearch:
    """Tests for Indexes search functionality."""

    @pytest.mark.skip(reason="Segfaults with usearch 2.21.0 - see module docstring")
    def test_indexes_search_single_query(self, two_saved_indexes):
        """Search with single query vector returns Matches."""
        indexes = Indexes(paths=two_saved_indexes, view=True)
        query = np.array([0, 2, 3, 4], dtype=np.uint8)
        results = indexes.search(query, count=3)
        assert len(results) <= 3

    @pytest.mark.skip(reason="Segfaults with usearch 2.21.0 - see module docstring")
    def test_indexes_search_batch_query(self, two_saved_indexes):
        """Search with batch query vectors returns BatchMatches."""
        indexes = Indexes(paths=two_saved_indexes, view=True)
        queries = np.array([[0, 2, 3, 4], [1, 2, 3, 4]], dtype=np.uint8)
        results = indexes.search(queries, count=3)
        assert len(results) == 2  # Two queries

    @pytest.mark.skip(reason="Segfaults with usearch 2.21.0 - see module docstring")
    def test_indexes_search_finds_vectors_across_shards(self, two_saved_indexes):
        """Search finds vectors that exist in different shards."""
        indexes = Indexes(paths=two_saved_indexes, view=True)
        # Query similar to vector in first shard (key=0)
        query1 = np.array([0, 2, 3, 4], dtype=np.uint8)
        results1 = indexes.search(query1, count=1)
        assert results1.keys[0] == 0  # Should find exact match

        # Query similar to vector in second shard (key=10)
        query2 = np.array([1, 2, 3, 4], dtype=np.uint8)
        results2 = indexes.search(query2, count=1)
        assert results2.keys[0] == 10  # Should find exact match

    @pytest.mark.skip(reason="Segfaults with usearch 2.21.0 - see module docstring")
    def test_indexes_search_aggregates_results(self, two_saved_indexes):
        """Search returns results aggregated from all shards."""
        indexes = Indexes(paths=two_saved_indexes, view=True)
        query = np.array([0, 2, 3, 4], dtype=np.uint8)
        results = indexes.search(query, count=4)
        # Should get results from both shards
        assert len(results) == 4
        # Results should be sorted by distance
        for i in range(len(results) - 1):
            assert results.distances[i] <= results.distances[i + 1]


class TestIndexesMerge:
    """Tests for Indexes merge functionality."""

    def test_indexes_merge_index_object(self, two_saved_indexes):
        """Indexes can merge an Index object."""
        # Load an index and merge into Indexes
        viewed = Index.restore(two_saved_indexes[0], view=True)
        indexes = Indexes()
        indexes._compiled.merge(viewed._compiled)
        initial_size = len(indexes)

        # Create another index and merge it
        idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
        idx.add(100, np.array([9, 9, 9, 9], dtype=np.uint8))
        indexes._compiled.merge(idx._compiled)

        assert len(indexes) == initial_size + 1

    @pytest.mark.skip(reason="Segfaults with usearch 2.21.0 - see module docstring")
    def test_indexes_merge_paths(self, two_saved_indexes, tmp_path):
        """Indexes can merge additional paths."""
        indexes = Indexes(paths=[two_saved_indexes[0]], view=True)
        initial_size = len(indexes)

        # Merge the second shard
        indexes._compiled.merge_paths([two_saved_indexes[1]], view=True, threads=0)
        assert len(indexes) == initial_size + 2  # Second shard has 2 vectors


class TestIndexesEdgeCases:
    """Tests for edge cases and potential issues."""

    def test_indexes_with_empty_index_file(self, tmp_path):
        """Indexes handles empty index files gracefully."""
        # Create and save an empty index
        idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
        path = tmp_path / "empty.usearch"
        idx.save(str(path))
        del idx

        indexes = Indexes(paths=[str(path)], view=True)
        assert len(indexes) == 0

    def test_indexes_search_empty_returns_empty_results(self, tmp_path):
        """Search on empty Indexes returns empty results."""
        idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
        path = tmp_path / "empty.usearch"
        idx.save(str(path))
        del idx

        indexes = Indexes(paths=[str(path)], view=True)
        query = np.array([0, 2, 3, 4], dtype=np.uint8)
        results = indexes.search(query, count=5)
        assert len(results) == 0

    @pytest.mark.skip(reason="Segfaults with usearch 2.21.0 - see module docstring")
    def test_indexes_sequential_creation_and_deletion(self, two_saved_indexes):
        """Creating and deleting Indexes objects sequentially works."""
        for _ in range(3):
            indexes = Indexes(paths=two_saved_indexes, view=True)
            assert len(indexes) == 4
            del indexes

    @pytest.mark.skip(reason="Segfaults with usearch 2.21.0 - see module docstring")
    def test_indexes_concurrent_instances(self, two_saved_indexes):
        """Multiple Indexes instances can coexist."""
        indexes1 = Indexes(paths=two_saved_indexes, view=True)
        indexes2 = Indexes(paths=two_saved_indexes, view=True)

        assert len(indexes1) == 4
        assert len(indexes2) == 4

        query = np.array([0, 2, 3, 4], dtype=np.uint8)
        results1 = indexes1.search(query, count=2)
        results2 = indexes2.search(query, count=2)

        assert len(results1) == len(results2)
