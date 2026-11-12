"""Two-layer causal attention: full-prefix recompute vs per-layer KV cache.

A small teaching experiment, not a real GPT benchmark. NumPy only.
Run: python kv_cache_equivalence.py
"""

from __future__ import annotations

import math
import numpy as np

SEED = 20261112
D_MODEL = 4
N_LAYERS = 2
VOCAB_SIZE = 10
MAX_POSITIONS = 32
PROMPT = [0, 1, 2, 3]
APPENDED_TOKENS = [4, 5, 6]  # predetermined continuation for the comparison

rng = np.random.default_rng(SEED)
embedding = rng.normal(0, 0.25, (VOCAB_SIZE, D_MODEL))
positions = rng.normal(0, 0.15, (MAX_POSITIONS, D_MODEL))
lm_head = rng.normal(0, 0.25, (D_MODEL, VOCAB_SIZE))
layers = []
for _ in range(N_LAYERS):
    layers.append({
        key: rng.normal(0, 0.25, (D_MODEL, D_MODEL))
        for key in ("wq", "wk", "wv", "wo", "w1", "w2")
    })


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)


def finish_layer(h: np.ndarray, attention: np.ndarray, weights: dict) -> np.ndarray:
    h = h + attention @ weights["wo"]   # attention residual
    return h + np.tanh(h @ weights["w1"]) @ weights["w2"]  # toy FFN residual


def full_forward(token_ids: list[int], return_cache: bool = False):
    """Process all positions, respecting the causal mask at every layer."""
    length = len(token_ids)
    h = embedding[token_ids] + positions[:length]
    cache = []
    for weights in layers:
        q, k, v = (h @ weights[name] for name in ("wq", "wk", "wv"))
        scores = q @ k.T / math.sqrt(D_MODEL)
        scores[np.triu_indices(length, k=1)] = -np.inf
        attention = softmax(scores) @ v
        cache.append((k.copy(), v.copy()))
        h = finish_layer(h, attention, weights)
    logits = h[-1] @ lm_head
    return (logits, cache) if return_cache else logits


def incremental_forward(token_id: int, cache: list[tuple[np.ndarray, np.ndarray]]):
    """Process only the newly appended token; reuse per-layer past keys/values."""
    old_length = cache[0][0].shape[0]
    h = embedding[[token_id]] + positions[old_length : old_length + 1]
    new_cache = []
    for weights, (past_k, past_v) in zip(layers, cache):
        q, k, v = (h @ weights[name] for name in ("wq", "wk", "wv"))
        all_k = np.concatenate((past_k, k), axis=0)
        all_v = np.concatenate((past_v, v), axis=0)
        # The query is the final position; all cached positions are earlier.
        attention = softmax(q @ all_k.T / math.sqrt(D_MODEL)) @ all_v
        new_cache.append((all_k, all_v))
        h = finish_layer(h, attention, weights)
    return h[-1] @ lm_head, new_cache


def run() -> None:
    prefix = PROMPT.copy()
    cached_logits, cache = full_forward(prefix, return_cache=True)  # prefill
    full_rows = 0
    cached_rows = 0
    full_attention_pairs = 0
    cached_attention_pairs = 0
    max_diff = 0.0
    for step in range(len(APPENDED_TOKENS) + 1):
        t = len(prefix)
        expected = full_forward(prefix)
        diff = float(np.max(np.abs(expected - cached_logits)))
        max_diff = max(max_diff, diff)
        assert np.allclose(expected, cached_logits, rtol=1e-12, atol=1e-12)
        print(f"step={step} context={t} logits_max_abs_diff={diff:.3e}")
        full_rows += t
        full_attention_pairs += t * (t + 1) // 2
        if step == 0:
            cached_rows += len(PROMPT)
            cached_attention_pairs += len(PROMPT) * (len(PROMPT) + 1) // 2
        else:
            cached_rows += 1
            cached_attention_pairs += t
        if step < len(APPENDED_TOKENS):
            next_token = APPENDED_TOKENS[step]
            prefix.append(next_token)
            cached_logits, cache = incremental_forward(next_token, cache)

    print(f"max_difference={max_diff:.3e}")
    print(f"token_rows_per_layer_full={full_rows}, cached={cached_rows}")
    print(f"valid_attention_pairs_per_layer_full={full_attention_pairs}, cached={cached_attention_pairs}")
    print(f"cached_tokens_at_end={len(prefix)}, per_layer_kv_entries={len(prefix) * 2}")
    print("PASSED: full recomputation and KV-cache incremental processing agree")


if __name__ == "__main__":
    run()
