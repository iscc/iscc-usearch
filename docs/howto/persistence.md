# Persistence

This guide covers saving, loading, and memory-mapping `NphdIndex` instances.

## Save an index

```python
index.save("my_index.usearch")
```

This writes the full index (HNSW graph + vectors) to a single file.

## Load an index

`load()` reads the entire file into RAM:

```python
from iscc_usearch import NphdIndex

index = NphdIndex()
index.load("my_index.usearch")
```

Or use the static `restore()` method to create and load in one step:

```python
index = NphdIndex.restore("my_index.usearch")
```

## Memory-map an index (view)

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

    A viewed index is **read-only**. Calling `add()` on a viewed index will fail.

## Restore (auto-detect)

`NphdIndex.restore()` dispatches to `load()` or `view()` based on the `view` parameter:

```python
# Full load (default)
index = NphdIndex.restore("my_index.usearch")

# Memory-mapped
index = NphdIndex.restore("my_index.usearch", view=True)
```

## Copy an index

Create an independent in-memory copy with the same configuration and data:

```python
copy = index.copy()
```

The copy is fully independent -- modifying one does not affect the other.

## When to use which

| Method      | RAM usage | Startup speed | Writable | Use case                       |
| ----------- | --------- | ------------- | -------- | ------------------------------ |
| `load()`    | High      | Slower        | Yes      | Read-write workloads           |
| `view()`    | Low       | Fast          | No       | Read-only serving, many shards |
| `restore()` | Either    | Either        | Either   | Convenience dispatcher         |
| `copy()`    | High      | Instant       | Yes      | Fork an index for experiments  |

## Metric restoration

When loading or viewing an index, `NphdIndex` automatically restores the custom NPHD metric.
USearch's native `load()`/`view()` replaces the compiled metric with the saved metric kind
(standard Hamming), so `NphdIndex` calls `change_metric()` after every load/view operation. This is
transparent -- you do not need to do anything special.
