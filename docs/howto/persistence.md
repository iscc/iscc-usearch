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

This serializes the index to an in-memory buffer, writes it to disk via `os.write()`,
flushes to stable storage with `fdatasync`, then atomically renames the temp file into place.
The result is both atomic (no partial files on crash) and durable (data survives power loss).

Sharded indexes use the same durable file-save path for `.usearch` shard files. `ShardedIndex.save()`
does not accept a path argument; it saves the active shard, bloom filter, and tombstones into the
directory configured at construction time:

```python
index.save()
```

Long-running sharded saves log start and completion messages at `INFO`, including the shard name
and vector count.

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

## Dirty counter

`NphdIndex` tracks unsaved mutations via the `dirty` property. It increments on each `add()` or
`remove()` call and resets to 0 on `save()`, `load()`, `view()`, and `reset()`:

```python
index = NphdIndex(max_dim=256)
index.add(1, vec)
print(index.dirty)  # 1

index.save("my_index.usearch")
print(index.dirty)  # 0
```

Use `dirty` to implement caller-driven flush policies (e.g., "save every N writes").

## Metric persistence

The native `MetricKind.NPHD` metric is correctly serialized and deserialized by usearch-iscc.
No manual metric restoration is needed after `load()` or `view()` operations.
