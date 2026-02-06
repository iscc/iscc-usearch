---
icon: lucide/house
---

# iscc-usearch

[![Tests](https://github.com/iscc/iscc-usearch/actions/workflows/tests.yml/badge.svg)](https://github.com/iscc/iscc-usearch/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/iscc/iscc-usearch/blob/main/LICENSE)

`iscc-usearch` is a thin wrapper around [USearch](https://github.com/unum-cloud/usearch) for
approximate nearest neighbor search (ANNS) over variable-length binary bit-vectors. The
[ISCC](https://iscc.codes) project needed prefix-compatible similarity search: shorter content
fingerprints are valid prefixes of longer ones, and both must be searchable in the same index.
Stock USearch has no support for this, so `iscc-usearch` adds a custom Normalized Prefix Hamming
Distance (NPHD) metric to make it work.

## Installation

```bash
pip install iscc-usearch
```

## Where to start

<div class="grid cards" markdown>

- **[Tutorials](tutorials/getting-started.md)** -- Learn the basics

    Hands-on guides from installation to working code.

- **[How-to guides](howto/persistence.md)** -- Solve specific problems

    Recipes for persistence, sharding, upsert, and bloom filters.

- **[Explanation](explanation/nphd-metric.md)** -- Understand the design

    Background on NPHD, architecture, sharding, and performance.

- **[Reference](reference/api.md)** -- API details

    Auto-generated API documentation for all public classes.

- **[Development](development/contributing.md)** -- Contribute

    Dev setup, testing, and contribution guidelines.

</div>
