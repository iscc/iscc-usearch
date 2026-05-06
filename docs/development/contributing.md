---
icon: lucide/git-pull-request
description: Developer setup with uv, testing with 100% coverage, linting, type checking, security scanning, and details on the patched USearch fork.
---

# Contributing

## Dev environment setup

1. Clone the repository:

    ```bash
    git clone https://github.com/iscc/iscc-usearch.git
    cd iscc-usearch
    ```

1. Install dependencies with [uv](https://docs.astral.sh/uv/):

    ```bash
    uv sync
    ```

    This installs the project in development mode with all dev dependencies.

## Running tests

Tests require 100% code coverage. To run the full suite:

```bash
uv run pytest
```

To run a single test file or function:

```bash
uv run pytest tests/test_nphd.py
uv run pytest tests/test_nphd.py::test_nphd_index_add_and_search
```

To generate a coverage report with missing lines:

```bash
uv run pytest --cov=iscc_usearch --cov-report=term-missing
```

## Linting and formatting

```bash
# Check for lint issues
uv run ruff check .

# Auto-format code
uv run ruff format .
```

The max line length is 119 characters and line endings are LF, both configured in `pyproject.toml`.

## Type checking

```bash
uv run ty check
```

Some usearch-related type errors are downgraded to warnings in `pyproject.toml` because usearch has
incomplete type annotations.

## Security scanning

```bash
uv run bandit -r src/
```

`assert` statements are allowed (`B101` is skipped) because the project uses them for parameter validation.

## Documentation

Serve the docs locally with live reload:

```bash
uv run poe docs-serve
```

Build for deployment:

```bash
uv run poe docs-build
```

## Cross-platform support

All code, scripts, and dev tools must work on **Linux**, **macOS**, and **Windows**. Please test on
multiple platforms when possible.

## Patched usearch fork

`iscc-usearch` depends on a [patched usearch fork](https://github.com/iscc/usearch), published on
PyPI as [`usearch-iscc`](https://pypi.org/project/usearch-iscc/) and installed automatically as a
regular dependency. The required version is declared in
[`pyproject.toml`](https://github.com/iscc/iscc-usearch/blob/main/pyproject.toml).

Each patch is maintained on a separate branch to facilitate upstream merging:

- [`fix-view-overhead`](https://github.com/iscc/usearch/compare/unum-cloud:main...fix-view-overhead) —
    Fast `view()` with computed offsets, lazy key reindexing, and GIL release for parallel shard loading
- [`patch-index-get`](https://github.com/iscc/usearch/compare/unum-cloud:main...patch-index-get) —
    Return `None` for non-existent keys in `Index.get()`
- [`fix-search-count-zero`](https://github.com/iscc/usearch/compare/unum-cloud:main...fix-search-count-zero) —
    Validate search count parameter to prevent segfault
- [`fix-serialized-length`](https://github.com/iscc/usearch/compare/unum-cloud:main...fix-serialized-length) —
    Fix `serialized_length` pybind11 default argument binding
- [`fix-array-copy-keyword`](https://github.com/iscc/usearch/compare/unum-cloud:main...fix-array-copy-keyword) —
    Accept `copy` keyword in `IndexedKeys.__array__` for NumPy 2.0
- [`python-128bit-keys`](https://github.com/iscc/usearch/compare/unum-cloud:main...python-128bit-keys) —
    Add 128-bit (UUID) key support to `Index` and `Indexes` via `key_kind="uuid"`
- [`add-nphd-metric`](https://github.com/iscc/usearch/compare/unum-cloud:main...add-nphd-metric) —
    Native `MetricKind.NPHD` for Normalized Prefix Hamming Distance
- [`fix-add-skip-duplicates`](https://github.com/iscc/usearch/compare/unum-cloud:main...fix-add-skip-duplicates) —
    Skip duplicate keys silently in `add()` instead of erroring with partial commit
    ([iscc/usearch#6](https://github.com/iscc/usearch/issues/6))

See the [Performance explanation](../explanation/performance.md) for details on the performance
patches.
