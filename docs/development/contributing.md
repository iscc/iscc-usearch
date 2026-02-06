---
icon: lucide/git-pull-request
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

`iscc-usearch` depends on a [patched usearch fork](https://github.com/iscc/usearch) (v2.23.3).
Pre-built wheels are hosted at <https://iscc.github.io/usearch/> and installed automatically via
platform-specific dependency specifiers in `pyproject.toml`.

The fork includes:

- Fast `view()` for memory-mapped indexes (computed offsets instead of pointer tables)
- Lazy key reindexing with double-checked locking
- GIL release in Python bindings for parallel shard loading
- Fixes for `get()` on missing keys, `search(count=0)`, and NumPy 2.0 compatibility

See the [Performance explanation](../explanation/performance.md) for details.
