---
icon: lucide/bot
description: Prescriptive reference for AI coding agents — constraints, decision rules, task recipes, and change playbooks.
---

# For Coding Agents

Dense, prescriptive reference for AI agents working on or integrating with `iscc-usearch`.
Tables and code over prose. Terminology matches the codebase exactly.

## Architecture map

### File layout

| File                               | Contains                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------- |
| `src/iscc_usearch/__init__.py`     | Public re-exports (`__all__`), package docstring                                      |
| `src/iscc_usearch/index.py`        | `Index` — USearch wrapper with upsert support (internal base)                         |
| `src/iscc_usearch/nphd.py`         | `NphdIndex`, `pad_vectors`, `unpad_vectors`, type aliases                             |
| `src/iscc_usearch/sharded.py`      | `ShardedIndex`, `ShardedIndex128`, `_UuidKeyMixin`, lazy iterators, bloom integration |
| `src/iscc_usearch/sharded_nphd.py` | `ShardedNphdIndex`, `ShardedNphdIndex128`, `ShardedNphdIndexedVectors`                |
| `src/iscc_usearch/bloom.py`        | `ScalableBloomFilter` — chained bloom filter with persistence                         |
| `src/iscc_usearch/utils.py`        | `timer`, `atomic_write`, `durable_write`                                              |

### Class hierarchy

```
usearch.index.Index
└── Index                            (index.py — adds upsert, internal base)
    └── NphdIndex                    (nphd.py — variable-length + NPHD metric)

ShardedIndex                         (sharded.py — composition-based sharding, uint64 keys)
├── ShardedIndex128                  (sharded.py — 128-bit UUID keys via _UuidKeyMixin)
└── ShardedNphdIndex                 (sharded_nphd.py — variable-length + NPHD metric)
    └── ShardedNphdIndex128          (sharded_nphd.py — 128-bit UUID keys via _UuidKeyMixin)

ScalableBloomFilter                  (bloom.py — standalone, used by ShardedIndex)
```

### Import dependency flow

```
utils.py
  ↑
bloom.py ← sharded.py ← sharded_nphd.py
              ↑
index.py ← nphd.py
```

### Public API exports (`__all__`)

```python
(
    "NphdIndex",
    "ShardedIndex",
    "ShardedIndex128",
)
(
    "ShardedNphdIndex",
    "ShardedNphdIndex128",
)
"ScalableBloomFilter", "timer"
```

`Index` is **not** exported — it is an internal base class.

---

## Decision dispatch

### Which index class?

| Use case                                              | Class                 | Key type  |
| ----------------------------------------------------- | --------------------- | --------- |
| Variable-length ISCC codes, fits in RAM               | `NphdIndex`           | uint64    |
| Fixed-length vectors, large scale                     | `ShardedIndex`        | uint64    |
| Fixed-length vectors, 128-bit UUID keys               | `ShardedIndex128`     | bytes(16) |
| Variable-length ISCC codes, large scale (production)  | `ShardedNphdIndex`    | uint64    |
| Variable-length ISCC codes, large scale, 128-bit keys | `ShardedNphdIndex128` | bytes(16) |

### Which persistence method?

| Method          | Reads file into RAM |     Memory-mapped      | Creates new object  |  Resets dirty counter   |
| --------------- | :-----------------: | :--------------------: | :-----------------: | :---------------------: |
| `load(path)`    |         Yes         |           No           |  No (mutates self)  |           Yes           |
| `view(path)`    |         No          |    Yes (read-only)     |  No (mutates self)  |           Yes           |
| `restore(path)` |    Yes (default)    | Optional (`view=True`) | Yes (static method) |           Yes           |
| `copy()`        |         N/A         |          N/A           |   Yes (deep copy)   | No (new index, dirty=0) |

`load`/`view`/`restore` are on `NphdIndex`. Sharded indexes handle persistence internally
via `save()`/constructor — they auto-load/view shards from the `path` directory.

### Which add method?

| Method       | Existing key behavior                                      | Duplicate keys in batch                             | keys=None | Requires multi=False |
| ------------ | ---------------------------------------------------------- | --------------------------------------------------- | :-------: | :------------------: |
| `add()`      | Skips duplicates in `multi=False`; appends in `multi=True` | Skipped in `multi=False`; all added in `multi=True` |    Yes    |          No          |
| `upsert()`   | Remove old + add new (last-write-wins)                     | Last occurrence kept                                |    No     |         Yes          |
| `add_once()` | Silently skipped across all shards                         | First occurrence kept                               |    No     |          No          |

For sharded indexes, `add()` only checks duplicate keys within the active shard. It does not scan
view shards. Use `add_once()` when keys may already exist across shard boundaries and first-write-wins
semantics are required.

---

## Constraints and invariants

### Concurrency rules

- **Single-process only.** No file locking on `.usearch` files.
- Multiple processes against the same index → **data corruption**.
- Within one process: `async`/`await` is safe for concurrent reads.
- Threads require an external lock — index objects are not thread-safe.
- Shard rotation is not atomic — concurrent writes must be serialized.

### Vector constraints

| Constraint            | Value                                                                 | Enforced in                                       |
| --------------------- | --------------------------------------------------------------------- | ------------------------------------------------- |
| `max_dim` upper bound | 256 (bits)                                                            | `NphdIndex.__init__`, `ShardedNphdIndex.__init__` |
| `max_dim` alignment   | Multiple of 8                                                         | `NphdIndex.__init__`                              |
| Vector dtype          | `np.uint8` (packed bits)                                              | `pad_vectors`                                     |
| Scalar kind           | `ScalarKind.B1`                                                       | `NphdIndex.__init__`                              |
| Internal ndim         | `max_dim + 8` (length byte = 8 bits)                                  | `NphdIndex.__init__`                              |
| Truncation            | Vectors longer than `max_bytes` are silently truncated to `max_bytes` | `pad_vectors`                                     |
| Length byte           | `padded[i, 0]` = original vector length in bytes                      | `pad_vectors` / `unpad_vectors`                   |

### Key constraints

| Variant          | Single key type  | Batch key type            | Tombstone canonical type |
| ---------------- | ---------------- | ------------------------- | ------------------------ |
| uint64 (default) | `int`            | `np.ndarray[np.uint64]`   | `int`                    |
| 128-bit UUID     | `bytes` (len 16) | `np.ndarray[dtype='V16']` | `bytes`                  |

- 128-bit variants: `keys=None` (auto-generation) raises `ValueError`.
- `_bloom_key()` / `_tombstone_key()` convert raw keys to canonical Python types.

### Dirty counter semantics

| Operation           | Effect on `dirty`                      |
| ------------------- | -------------------------------------- |
| `add()`             | `+= count_added`                       |
| `remove()`          | `+= count_of_existing_keys_removed`    |
| `upsert()`          | Combines remove + add effects          |
| `save()`            | Reset to 0                             |
| `load()` / `view()` | Reset to 0                             |
| `reset()`           | Reset to 0                             |
| `compact()`         | Reset to 0 (calls `save()` internally) |
| Shard rotation      | **Not** reset — survives rotation      |
| `read_only=True`    | Always returns 0                       |

### Bloom filter behavior

- Enabled by default (`bloom_filter=True` on `ShardedIndex`).
- Updated on `add()` — keys are added via `add_batch()`.
- **Never updated on `remove()`** — bloom filter is append-only.
- `contains()` / `get()` / `remove()` use bloom for fast rejection of non-existent keys.
- False positives are expected (probabilistic). False negatives do not occur.
- `rebuild_bloom()` creates a fresh filter from all existing keys.
- Automatically rebuilt on load if the file is missing or corrupt (writable indexes only).
- `compact()` calls `rebuild_bloom()` internally.
- Bloom filter file: `bloom.isbf` in the shard directory.
- Always loaded from disk if file exists, regardless of `_use_bloom` setting.

### Tombstone lifecycle

```
remove(key) → key added to _tombstones (in-memory set)
    ↓
save() → bloom persisted, then shard, then _tombstones as tombstones.npy
    ↓
compact() → view shards rebuilt excluding tombstoned entries
    ↓
_tombstones.clear(), tombstones.npy deleted
```

- `_needs_compact` flag: set when `add()` clears a tombstone (creating cross-shard duplicates).
- `tombstones.npy` file existence signals `_needs_compact=True` on next load.
- Tombstones are never written to bloom filter — bloom has no "remove" operation.
- Tombstones are persisted **after** shard data so that tombstone removals only become
    visible once the shard they depend on is durable.

---

## Side effects catalog

| Method            | Disk writes                                                             | `_dirty`                | `_tombstones`              | Bloom update         | Shard rotation         |
| ----------------- | ----------------------------------------------------------------------- | ----------------------- | -------------------------- | -------------------- | ---------------------- |
| `add()`           | None (until rotation; rotation durably writes bloom, shard, tombstones) | `+= count_added`        | Clears matching tombstones | `add_batch()`        | Yes (if size exceeded) |
| `remove()`        | None                                                                    | `+= N` (existing only)  | Adds view-shard keys       | None                 | No                     |
| `upsert()`        | None                                                                    | Via remove + add        | Via remove + add           | Via add              | Via add                |
| `add_once()`      | None                                                                    | Via add (new keys only) | None                       | Via add              | Via add                |
| `save()`          | `.usearch`, `bloom.isbf`, `tombstones.npy`                              | Reset to 0              | Persisted                  | Persisted            | No                     |
| `load()`          | None                                                                    | Reset to 0              | Loaded from `.npy`         | Loaded from `.isbf`  | No                     |
| `view()`          | None                                                                    | Reset to 0              | N/A (NphdIndex)            | N/A                  | No                     |
| `reset()`         | None                                                                    | Reset to 0              | Cleared                    | Cleared              | No                     |
| `compact()`       | Rebuilds `.usearch`, `bloom.isbf`, deletes `tombstones.npy`             | Reset to 0              | Cleared                    | Rebuilt              | No                     |
| `search()`        | None                                                                    | No change               | No change                  | None                 | No                     |
| `get()`           | None                                                                    | No change               | No change                  | None                 | No                     |
| `rebuild_bloom()` | `bloom.isbf` (if `save=True`)                                           | No change               | No change                  | Rebuilt from scratch | No                     |

---

## Task recipes

### Add vectors to a sharded index

```python
from iscc_usearch import ShardedNphdIndex
import numpy as np

idx = ShardedNphdIndex(max_dim=256, path="my_index")

# Single vector
key = 1
vec = np.random.randint(0, 256, size=32, dtype=np.uint8)
idx.add(key, vec)

# Batch of uniform-length vectors
keys = np.arange(100, 200, dtype=np.uint64)
vecs = np.random.randint(0, 256, size=(100, 32), dtype=np.uint8)
idx.add(keys, vecs)

# Batch of variable-length vectors
keys2 = np.arange(200, 205, dtype=np.uint64)
vecs2 = [np.random.randint(0, 256, size=s, dtype=np.uint8) for s in [8, 16, 24, 32, 8]]
idx.add(keys2, vecs2)

idx.save()
```

### Search with mixed bit-lengths

```python
# Query vectors can differ in length from stored vectors.
# NPHD metric normalizes by the shorter prefix.
q8 = np.random.randint(0, 256, size=8, dtype=np.uint8)  # 64-bit query
q32 = np.random.randint(0, 256, size=32, dtype=np.uint8)  # 256-bit query

matches_8 = idx.search(q8, count=10)  # compares prefix of stored vectors
matches_32 = idx.search(q32, count=10)  # full comparison with 256-bit vectors

# Batch search with variable-length queries
queries = [q8, q32]
batch_matches = idx.search(queries, count=5)
```

### Flush based on dirty counter

```python
FLUSH_THRESHOLD = 10_000

idx.add(keys, vecs)
if idx.dirty >= FLUSH_THRESHOLD:
    idx.save()
```

### Upsert then compact

```python
# Upsert replaces existing vectors (remove + add).
# After many upserts, view shards accumulate stale entries.
idx.upsert(key, new_vec)

# Compact rebuilds view shards to reclaim space.
# Call periodically, not after every upsert.
if idx.tombstone_count > 1000:
    removed = idx.compact()
```

### Rebuild bloom filter for existing index

```python
# For indexes created without bloom filter, or after corruption:
idx = ShardedNphdIndex(max_dim=256, path="existing_index")
count = idx.rebuild_bloom()  # Scans all shards, saves bloom.isbf
```

### Open existing index read-only

```python
# All shards memory-mapped, no active shard. Write operations raise RuntimeError.
idx = ShardedNphdIndex(path="existing_index", read_only=True)
matches = idx.search(query_vec)
# idx.add(...)  # RuntimeError: index is read-only
```

---

## Change playbook

### If modifying padding logic

Files: `nphd.py` (`pad_vectors`, `unpad_vectors`)

Also update:

- `sharded_nphd.py` — `add()`, `search()`, `get()`, `vectors` iterator (all call `pad_vectors`/`unpad_vectors`)
- `tests/test_nphd.py` — padding round-trip tests
- `docs/explanation/architecture.md` — "Length-prefixed padding" section

### If modifying shard rotation

Files: `sharded.py` (`_rotate_shard`, `_schedule_next_size_check`, `_adds_until_size_check`)

Also update:

- `sharded.py` — `add()` (calls rotation), `_load_existing()` (loads rotated shards)
- `sharded_nphd.py` — `_create_shard()`, `_restore_shard()` (shard creation/restore hooks)
- `docs/explanation/sharding-design.md`
- `docs/howto/sharding.md`

### If modifying bloom filter format

Files: `bloom.py` (`save`, `load`, `MAGIC`, `VERSION`)

Also update:

- `sharded.py` — `_load_bloom_if_exists()`, `save()`, `rebuild_bloom()`, `compact()`
- Bump `VERSION` constant in `bloom.py` if format changes
- `tests/test_bloom.py`
- `docs/howto/bloom-filters.md`

### If modifying key handling

Files: `sharded.py` (`_UuidKeyMixin`, key hook methods on `ShardedIndex`)

Also update:

- `sharded_nphd.py` — `ShardedNphdIndex128` inherits `_UuidKeyMixin`
- `index.py` — `Index.upsert()` has its own key normalization
- `tests/test_sharded.py`, `tests/test_sharded_nphd.py` — key-related tests
- `docs/howto/uuid-keys.md`

### If adding a new index class

Also update:

- `src/iscc_usearch/__init__.py` — add to `__all__` and import
- `docs/reference/api.md` — add mkdocstrings directive (or handled via `gen_llms_full.py`)
- `scripts/gen_llms_full.py` — add to `_API_CLASSES` list
- `docs/explanation/architecture.md` — update class hierarchy and "Choosing an index class" table
- This page — update architecture map, class hierarchy, and decision dispatch tables
- Add test file in `tests/`

---

## Common mistakes

**NEVER** open the same index directory from multiple processes.
Use a single process with `async`/`await` for concurrent access.

---

**NEVER** mutate an index opened with `view()`. Viewed indexes are memory-mapped and read-only.
Use `load()` or `restore()` for mutable access. Sharded indexes use `read_only=True` for the
same purpose.

---

**NEVER** pass `ndim`, `metric`, or `dtype` to `NphdIndex`. These are computed from `max_dim`.
Passing them raises `TypeError`.

```python
# WRONG
idx = NphdIndex(ndim=264, metric=MetricKind.NPHD)

# CORRECT
idx = NphdIndex(max_dim=256)
```

---

**NEVER** skip `compact()` after sustained upsert/remove workloads. View shards accumulate
tombstoned and duplicate entries that waste disk space and slow searches.

---

**NEVER** assume `len(sharded_index)` is exact after upserts without compaction. Cross-shard
duplicates cause `size` to overcount. Call `compact()` for an exact count.

---

**ALWAYS** call `save()` before process exit. Unsaved data in the active shard is lost.
Use the `dirty` counter to implement flush policies.

---

**ALWAYS** use `np.uint8` dtype for vectors. Other dtypes cause silent data corruption in
the NPHD metric computation.

---

**ALWAYS** pass explicit keys to `upsert()` and `add_once()`. Both raise `ValueError` on
`keys=None`. For 128-bit variants, `add()` also requires explicit keys.

---

**ALWAYS** serialize concurrent writes within a single process (e.g., via `asyncio.Lock`).
Shard rotation during concurrent writes can cause data loss.
