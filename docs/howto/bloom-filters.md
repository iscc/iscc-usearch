---
icon: lucide/filter
---

# Bloom filters

`ScalableBloomFilter` is a standalone probabilistic data structure for fast key existence checks.

## Overview

A bloom filter answers the question "is this element in the set?" with one of two results:

- **Definitely not** (guaranteed correct)
- **Probably yes** (small chance of false positive)

`ScalableBloomFilter` chains multiple fixed-size bloom filters so the set can grow without bound
while the false positive rate stays controlled.

## Create a bloom filter

```python
from iscc_usearch import ScalableBloomFilter

bloom = ScalableBloomFilter(
    initial_capacity=1_000_000,  # Expected elements before first growth
    fpr=0.01,  # 1% target false positive rate
)
```

## Add elements

```python
# Single element
bloom.add(42)

# Batch (more efficient for large sets)
bloom.add_batch([100, 200, 300, 400, 500])
```

## Check membership

```python
print(bloom.contains(42))  # True
print(bloom.contains(999))  # False (definitely not present)

# Python 'in' operator works too
print(42 in bloom)  # True

# Batch check
results = bloom.contains_batch([42, 999, 100])
print(results)  # [True, False, True]
```

## Inspect state

```python
print(bloom.count)  # Approximate number of elements added
print(bloom.current_capacity)  # Total capacity across all filters
print(bloom.filter_count)  # Number of filters in the chain
print(len(bloom))  # Same as bloom.count
```

## Save and load

```python
# Save to disk
bloom.save("my_bloom.isbf")

# Load later
bloom = ScalableBloomFilter.load("my_bloom.isbf")
```

The `.isbf` file format stores all filter state including capacities and hash counts.

## Reset

```python
bloom.clear()  # Reset to initial state (empty, single filter)
```

## Scaling behavior

When the current filter reaches capacity, a new filter is added to the chain automatically.
Each new filter has **double** the capacity of the previous one (configurable via `growth_factor`)
and a **tighter** false positive rate (geometric series with r=0.5) to keep the overall FPR
controlled. Membership checks search the newest filter first since recent keys are most likely
there.

## Choosing parameters

| Parameter          | Default    | Guidance                                                 |
| ------------------ | ---------- | -------------------------------------------------------- |
| `initial_capacity` | 10,000,000 | Set to expected dataset size to minimize filter chaining |
| `fpr`              | 0.01       | Lower = fewer false positives, more memory per element   |
| `growth_factor`    | 2.0        | Higher = fewer filters, larger memory jumps              |

## Integration with sharded indexes

`ShardedIndex` and `ShardedNphdIndex` use `ScalableBloomFilter` internally (enabled by default via
`bloom_filter=True`) for O(1) rejection of non-existent keys in `get()`, `contains()`, and
`count()`. There is no need to manage the filter manually when using sharded indexes.
