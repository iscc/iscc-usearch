---
icon: lucide/hard-drive
description: Save, load, memory-map, and restore NphdIndex instances. Compare load, view, restore, and copy methods for different RAM and startup trade-offs.
---

# Persistence

This guide shows how to save, load, and memory-map `NphdIndex` instances.

## Save an index to disk

```python
index.save("my_index.usearch")
```

This writes the full index (HNSW graph and vectors) to a single file.

## Load an index from disk

`load()` reads the entire file into RAM:

```python
from iscc_usearch import NphdIndex

index = NphdIndex()
index.load("my_index.usearch")
```

You can also use `restore()` to create and load in one step:

```python
index = NphdIndex.restore("my_index.usearch")
```

## Memory-map an index

`view()` memory-maps the file for read-only access. The OS pages data in on demand, so startup is
fast and memory usage stays low:

```python
index = NphdIndex.restore("my_index.usearch", view=True)
```

Or explicitly:

```python
index = NphdIndex()
index.view("my_index.usearch")
```

!!! warning

    A viewed index is **read-only**. Calling `add()` on a viewed index raises an error from
    USearch's C++ core.

## Restore with auto-detect

`NphdIndex.restore()` calls either `load()` or `view()` based on the `view` parameter:

```python
# Full load (default)
index = NphdIndex.restore("my_index.usearch")

# Memory-mapped
index = NphdIndex.restore("my_index.usearch", view=True)
```

## Copy an index

`copy()` creates an independent in-memory clone with the same configuration and data:

```python
copy = index.copy()
```

The copy is independent. Modifying one does not affect the other.

## Choosing a method

| Method      | RAM usage | Startup speed | Writable | Use case                       |
| ----------- | --------- | ------------- | -------- | ------------------------------ |
| `load()`    | High      | Slower        | Yes      | Read-write workloads           |
| `view()`    | Low       | Fast          | No       | Read-only serving, many shards |
| `restore()` | Either    | Either        | Either   | Convenience dispatcher         |
| `copy()`    | High      | Instant       | Yes      | Fork an index for experiments  |

## Metric restoration

`NphdIndex` automatically restores the custom NPHD metric after loading or viewing an index.
USearch's native `load()` and `view()` replace the compiled metric with standard Hamming, so
`NphdIndex` calls `change_metric()` after every load or view operation. This is handled for you
-- no extra steps required.
