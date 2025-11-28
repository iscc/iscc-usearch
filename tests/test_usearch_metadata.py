"""
Confirm the expected behavior of usearch Index.metadata() with

- metric=MetricKind.Hamming
- dtype=ScalarKind.B1
- Reading metadata from file path
- Reading metadata from buffer
"""

import numpy as np
from usearch.index import Index, MetricKind, ScalarKind


def test_metadata_from_path_returns_dict(tmp_path):
    """Index.metadata(path) returns dict."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    file_path = tmp_path / "test.usearch"
    idx.save(str(file_path))

    result = Index.metadata(str(file_path))

    assert isinstance(result, dict)


def test_metadata_from_buffer_returns_dict():
    """Index.metadata(buffer) returns dict."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    buffer = idx.save()

    result = Index.metadata(buffer)

    assert isinstance(result, dict)


def test_metadata_contains_dimensions(tmp_path):
    """Metadata dict contains 'dimensions' key with correct value."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    file_path = tmp_path / "test.usearch"
    idx.save(str(file_path))

    metadata = Index.metadata(str(file_path))

    assert "dimensions" in metadata
    assert metadata["dimensions"] == 32


def test_metadata_contains_scalar_kind(tmp_path):
    """Metadata dict contains 'kind_scalar' key."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    file_path = tmp_path / "test.usearch"
    idx.save(str(file_path))

    metadata = Index.metadata(str(file_path))

    assert "kind_scalar" in metadata
    assert metadata["kind_scalar"] == ScalarKind.B1


def test_metadata_contains_metric_kind(tmp_path):
    """Metadata dict contains 'kind_metric' key."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    file_path = tmp_path / "test.usearch"
    idx.save(str(file_path))

    metadata = Index.metadata(str(file_path))

    assert "kind_metric" in metadata
    assert metadata["kind_metric"] == MetricKind.Hamming


def test_metadata_nonexistent_file_returns_none(tmp_path):
    """Index.metadata('nonexistent.usearch') returns None."""
    nonexistent_path = tmp_path / "nonexistent.usearch"

    result = Index.metadata(str(nonexistent_path))

    assert result is None


def test_metadata_empty_index(tmp_path):
    """Metadata from empty saved index still has correct config."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)

    file_path = tmp_path / "empty.usearch"
    idx.save(str(file_path))

    metadata = Index.metadata(str(file_path))

    assert metadata is not None
    assert metadata["dimensions"] == 32
    assert metadata["kind_metric"] == MetricKind.Hamming
    assert metadata["kind_scalar"] == ScalarKind.B1


def test_metadata_keys(tmp_path):
    """Verify all expected keys exist in metadata dict."""
    idx = Index(ndim=32, metric=MetricKind.Hamming, dtype=ScalarKind.B1)
    idx.add(1, np.array([178, 204, 60, 240], dtype=np.uint8))

    file_path = tmp_path / "test.usearch"
    idx.save(str(file_path))

    metadata = Index.metadata(str(file_path))

    expected_keys = ["dimensions", "kind_scalar", "kind_metric"]
    for key in expected_keys:
        assert key in metadata
