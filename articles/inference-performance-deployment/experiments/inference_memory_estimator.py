"""Estimate LLM weight and KV-cache memory, plus an idealized decode traffic model.

This is a deterministic capacity/traffic calculator, not a GPU benchmark.
Run: python inference_memory_estimator.py
"""

from __future__ import annotations

from dataclasses import dataclass

KIB = 1024
GIB = 1024**3


@dataclass(frozen=True)
class ModelConfig:
    parameters: int
    weight_bytes_per_parameter: int
    layers: int
    kv_heads: int
    head_dim: int
    kv_bytes_per_element: int


def weight_memory_bytes(config: ModelConfig) -> int:
    return config.parameters * config.weight_bytes_per_parameter


def kv_bytes_per_token(config: ModelConfig) -> int:
    # One K and one V vector for every layer and KV head.
    return (
        2
        * config.layers
        * config.kv_heads
        * config.head_dim
        * config.kv_bytes_per_element
    )


def kv_cache_bytes(config: ModelConfig, context_lengths: list[int]) -> int:
    # Each request can have a different context length.
    return kv_bytes_per_token(config) * sum(context_lengths)


def gib(value: int | float) -> float:
    return value / GIB


def main() -> None:
    config = ModelConfig(
        parameters=8_000_000_000,
        weight_bytes_per_parameter=2,  # FP16
        layers=32,
        kv_heads=8,
        head_dim=128,
        kv_bytes_per_element=2,  # FP16
    )

    weights = weight_memory_bytes(config)
    per_token = kv_bytes_per_token(config)
    one_request = kv_cache_bytes(config, [8192])
    eight_requests = kv_cache_bytes(config, [8192] * 8)
    mixed_lengths = kv_cache_bytes(config, [8192, 4096])

    assert per_token == 128 * KIB
    assert one_request == GIB
    assert eight_requests == 8 * GIB
    assert mixed_lengths == int(1.5 * GIB)
    assert weights == 16_000_000_000

    print(f"KV per token per request: {per_token:,} bytes = {per_token / KIB:.0f} KiB")
    print(f"KV for 1 request x 8,192 tokens: {gib(one_request):.2f} GiB")
    print(f"KV for 8 requests x 8,192 tokens: {gib(eight_requests):.2f} GiB")
    print(f"KV for request lengths 8,192 + 4,096: {gib(mixed_lengths):.2f} GiB")
    print(
        f"FP16 weights: {weights:,} bytes = {weights / 1e9:.2f} GB "
        f"= {gib(weights):.2f} GiB"
    )
    total = weights + eight_requests
    print(f"Weights + KV for 8 requests: {gib(total):.2f} GiB")
    print(
        "Remaining on a hypothetical 24 GiB GPU before runtime/workspace: "
        f"{24 - gib(total):.2f} GiB"
    )

    # Idealized Decode assumption: read the weight matrix once per iteration
    # and reuse it across all active sequences in that iteration.
    print("\nIdealized weight traffic per generated token:")
    for batch_size in (1, 2, 4, 8):
        print(
            f"  batch={batch_size}: {weights / batch_size / 1e9:.2f} GB/token "
            "(weight bytes amortized across the batch)"
        )

    # This deliberately does not predict real latency: it excludes KV reads,
    # compute time, kernel overhead, cache effects, and scheduler/network costs.
    assumed_bandwidth = 2_000_000_000_000  # hypothetical 2 TB/s
    weight_only_step_seconds = weights / assumed_bandwidth
    print(
        "\nWeight-only theoretical lower bound at 2 TB/s: "
        f"{weight_only_step_seconds * 1000:.2f} ms per Decode iteration"
    )
    print("Not a measured GPU result; real Decode also reads KV state and performs computation.")


if __name__ == "__main__":
    main()
