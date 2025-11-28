"""
Custom metrics for ISCC vector database.

This module provides custom distance metrics for binary vectors, specifically
the Normalized Prefix Hamming Distance (NPHD) for variable-length ISCC codes.
"""

from numba import carray, cfunc, types
from usearch.index import CompiledMetric, MetricKind, MetricSignature

# Maximum supported vector size: 264 bits (33 bytes including length signal)
MAX_BYTES = 33

# Define the Numba signature for binary vectors
# We use uint8 (unsigned char) for binary data
NPHD_SIGNATURE = types.float32(
    types.CPointer(types.uint8),
    types.CPointer(types.uint8),
)


@cfunc(NPHD_SIGNATURE)
def nphd_distance(a, b):  # pragma: no cover
    # type: (types.CPointer[types.uint8], types.CPointer[types.uint8]) -> types.float32
    """
    Calculate NPHD between two variable-length binary vectors.

    Uses MAX_BYTES constant for buffer size.
    """
    a_array = carray(a, MAX_BYTES)
    b_array = carray(b, MAX_BYTES)

    # Extract length from first byte (1-255 bytes)
    a_bytes = types.uint8(a_array[0])
    b_bytes = types.uint8(b_array[0])

    # Use the shorter length for comparison (prefix compatibility)
    min_bytes = min(a_bytes, b_bytes)

    # Calculate Hamming distance over the common prefix
    # Start from byte 1 (skip length signal byte)
    hamming_distance = types.uint16(0)
    for byte_idx in range(1, min_bytes + 1):
        # XOR the bytes and count differing bits
        xor_result = types.uint8(a_array[byte_idx] ^ b_array[byte_idx])
        # Count set bits using Brian Kernighan's algorithm
        while xor_result > 0:
            hamming_distance += 1
            xor_result = types.uint8(xor_result & (xor_result - 1))

    # Convert to bits for normalization
    min_bits = min_bytes * 8

    # Normalize by the length of the shorter vector
    if min_bits == 0:
        return types.float32(0.0)

    normalized_distance = types.float32(hamming_distance) / types.float32(min_bits)
    return normalized_distance


def create_nphd_metric():
    # type: () -> CompiledMetric
    """
    Create a Normalized Prefix Hamming Distance (NPHD) metric for variable-length binary vectors.

    NPHD is designed for prefix-compatible codes where shorter codes are prefixes of longer codes.
    It normalizes the Hamming distance by the length of the common prefix (shorter vector).

    The metric expects binary vectors with a length signal in the first byte:
    - First byte: length signal (1-255) indicating actual vector length in bytes
    - Remaining bytes: actual binary vector data

    Uses a fixed maximum of 264 bits (33 bytes) for vector storage.

    :return: CompiledMetric instance for use with usearch Index
    """
    # Create and return the compiled metric
    return CompiledMetric(
        pointer=nphd_distance.address,
        kind=MetricKind.Hamming,  # Use Hamming as base kind
        signature=MetricSignature.ArrayArray,
    )
