# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`iscc-usearch` is a thin wrapper around USearch providing scalable approximate nearest neighbor search (ANNS)
for variable-length binary bit-vectors. The core feature is the Normalized Prefix Hamming Distance (NPHD)
metric, which enables prefix-compatible similarity search where shorter codes are prefixes of longer codes.

## Commands

```bash
# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/test_nphd.py

# Run single test function
uv run pytest tests/test_nphd.py::test_nphd_index_add_and_search

# Run tests with coverage
uv run pytest --cov=iscc_usearch

# Type checking
uv run ty check

# Linting
uv run ruff check .

# Format code
uv run ruff format .

# Security scan
uv run bandit -r src/
```

## Architecture

### Core Components

- **`src/iscc_usearch/nphd.py`**: Main `NphdIndex` class extending USearch's `Index`. Handles variable-length
    vectors through length-prefixed padding. Key methods: `add()`, `search()`, `get()`, `load()`, `view()`,
    `restore()`, `copy()`.

- **`src/iscc_usearch/metrics.py`**: Custom NPHD distance metric compiled via Numba's `@cfunc`. Uses first byte
    as length signal, calculates Hamming distance over common prefix, normalizes by shorter vector length.

### Key Design Patterns

- **Length-prefixed padding**: Variable-length vectors are stored with first byte containing length, padded to
    uniform size (`pad_vectors`/`unpad_vectors` in nphd.py)
- **Custom metric restoration**: After `load()`/`view()`, the NPHD metric must be explicitly restored since
    USearch replaces it with standard Hamming
- **Single-process concurrency model**: No file locking - use async/await within single process for concurrent
    access

### Type Aliases (nphd.py)

```python
Key = int | None
Keys = Sequence[int] | None
Vector = NDArray[np.uint8]
Vectors = Sequence[NDArray[np.uint8]] | NDArray[np.uint8]
```

## Dependencies

- **usearch**: Core vector index library
- **numba**: JIT compilation for custom distance metric and padding functions
- **numpy**: Array operations
- **loguru**: Logging

## Deepwiki Resources

Use Deepwiki for help and exploration of the following libraries if you have knowledge gaps or questions:

**Dependencies:**

- unum-cloud/usearch
- numba/numba
- Delgan/loguru

**Development Dependencies:**

- nat-n/poethepoet
- PyCQA/bandit
- j178/prek
- astral-sh/ty
- pytest-dev/pytest
- pytest-dev/pytest-cov
- astral-sh/ruff
