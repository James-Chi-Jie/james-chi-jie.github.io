"""Calculate request latency and system output throughput from toy timestamps.

The timestamps are deliberately constructed for teaching. This is not a GPU
benchmark and does not model a serving engine.
Run: python inference_metrics_example.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose


@dataclass(frozen=True)
class RequestTrace:
    request_id: str
    submitted_at: float
    token_times: tuple[float, ...]


def request_metrics(trace: RequestTrace) -> dict[str, float | int]:
    if not trace.token_times:
        raise ValueError("A request must contain at least one output token.")
    if any(
        later < earlier
        for earlier, later in zip(trace.token_times, trace.token_times[1:])
    ):
        raise ValueError("Token timestamps must be non-decreasing.")

    first_token = trace.token_times[0]
    last_token = trace.token_times[-1]
    token_count = len(trace.token_times)

    ttft = first_token - trace.submitted_at
    e2e = last_token - trace.submitted_at
    # TPOT is the mean interval between output tokens after the first one.
    # A one-token response has no inter-token interval, so TPOT is undefined.
    tpot = (
        (last_token - first_token) / (token_count - 1)
        if token_count > 1
        else float("nan")
    )

    return {
        "output_tokens": token_count,
        "ttft_seconds": ttft,
        "tpot_seconds": tpot,
        "e2e_seconds": e2e,
    }


def system_tps(traces: list[RequestTrace]) -> float:
    if not traces:
        raise ValueError("At least one request is required.")
    if any(not trace.token_times for trace in traces):
        raise ValueError("Every request must contain at least one output token.")

    first_request = min(trace.submitted_at for trace in traces)
    last_response = max(trace.token_times[-1] for trace in traces)
    total_output_tokens = sum(len(trace.token_times) for trace in traces)

    # This example uses the first-request-to-last-response measurement window.
    return total_output_tokens / (last_response - first_request)


def main() -> None:
    token_times = (0.2, 0.8, 1.4, 2.0)
    traces = [
        RequestTrace("A", 0.0, token_times),
        RequestTrace("B", 0.0, token_times),
        RequestTrace("C", 0.0, token_times),
    ]

    for trace in traces:
        metrics = request_metrics(trace)
        print(
            f"{trace.request_id}: "
            f"TTFT={metrics['ttft_seconds']:.2f}s, "
            f"TPOT={metrics['tpot_seconds']:.2f}s/token, "
            f"E2E={metrics['e2e_seconds']:.2f}s, "
            f"output={metrics['output_tokens']} tokens"
        )

    total_tokens = sum(len(trace.token_times) for trace in traces)
    system_rate = system_tps(traces)
    per_user_rate = len(token_times) / (
        token_times[-1] - traces[0].submitted_at
    )

    print(f"System TPS: {total_tokens} / 2.00s = {system_rate:.2f} output tokens/s")
    print(f"Per-request output rate: 4 / 2.00s = {per_user_rate:.2f} output tokens/s")

    assert isclose(request_metrics(traces[0])["ttft_seconds"], 0.2)
    assert isclose(request_metrics(traces[0])["tpot_seconds"], 0.6)
    assert isclose(request_metrics(traces[0])["e2e_seconds"], 2.0)
    assert isclose(system_rate, 6.0)
    print("PASSED: request metrics and system throughput match the constructed timeline.")


if __name__ == "__main__":
    main()
