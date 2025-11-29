"""Pytest fixtures for iscc_usearch tests."""

import pytest

from iscc_usearch import NphdIndex, ShardedNphdIndex


@pytest.fixture(params=["nphd", "sharded_nphd"])
def nphd_index_factory(request, tmp_path):
    """Factory fixture for creating NphdIndex or ShardedNphdIndex.

    Returns a callable: factory(max_dim=256, **kwargs) -> index

    Usage in tests:
        def test_something(nphd_index_factory):
            index = nphd_index_factory(max_dim=64)
            # test runs twice: once with NphdIndex, once with ShardedNphdIndex
    """
    if request.param == "nphd":

        def factory(max_dim=256, **kwargs):
            return NphdIndex(max_dim=max_dim, **kwargs)

        return factory
    else:
        # ShardedNphdIndex requires path, use tiny shard_size to test rotation
        counter = [0]

        def factory(max_dim=256, **kwargs):
            counter[0] += 1
            path = tmp_path / f"shards_{counter[0]}"
            return ShardedNphdIndex(max_dim=max_dim, path=path, shard_size=1024, **kwargs)

        return factory
