"""AI04: shifted targets, causal masking, and autoregressive steps.

A randomly initialized GPT-style miniature model, NOT a trained LLM.
Requires PyTorch. Run: python experiments/next_token_mechanics.py
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class TinyCausalReadout(nn.Module):
    """One causal self-attention and vocab head, not a complete GPT block."""

    def __init__(self, vocab_size=8, dim=8, max_length=16):
        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.position = nn.Embedding(max_length, dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads=2, batch_first=True, dropout=0.0
        )
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, token_ids):
        _, length = token_ids.shape
        positions = torch.arange(length, device=token_ids.device)
        x = self.token(token_ids) + self.position(positions)
        # True = future position is not accessible to the current position.
        future_mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        context, _ = self.attention(
            x, x, x, attn_mask=future_mask, need_weights=False
        )
        return self.output(x + context)  # [batch, positions, vocabulary]


def softmax_manual(values):
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    return [weight / sum(weights) for weight in weights]


def main():
    torch.manual_seed(7)
    model = TinyCausalReadout().eval()

    # Known training text: predict the next ID at each known prefix.
    text_ids = torch.tensor([[0, 1, 2, 3]])
    inputs, targets = text_ids[:, :-1], text_ids[:, 1:]
    logits = model(inputs)
    assert logits.shape == (1, 3, 8)
    assert targets.shape == (1, 3)
    loss = F.cross_entropy(logits.reshape(-1, 8), targets.reshape(-1))
    assert torch.isfinite(loss)

    # Alter the future: the first two outputs must remain identical.
    changed = torch.tensor([[0, 1, 6]])
    with torch.no_grad():
        original_scores = model(inputs)
        changed_scores = model(changed)
    assert torch.allclose(original_scores[:, :2], changed_scores[:, :2], atol=1e-6)
    assert not torch.allclose(original_scores[:, 2], changed_scores[:, 2], atol=1e-6)

    # The 3-class hand example is separate from randomly initialized weights.
    probabilities = softmax_manual([2.0, 1.0, 0.0])
    assert abs(sum(probabilities) - 1.0) < 1e-12
    assert max(range(3), key=lambda i: probabilities[i]) == 0

    # During generation, only the last position is used to choose a new ID.
    current = torch.tensor([[0, 1]])
    generated = []
    with torch.no_grad():
        for _ in range(2):
            next_scores = model(current)[:, -1, :]
            next_id = next_scores.argmax(dim=-1, keepdim=True)
            generated.append(int(next_id.item()))
            current = torch.cat((current, next_id), dim=1)
    assert current.shape[1] == 4
    assert current[0, 2:].tolist() == generated

    print("training inputs:  ", inputs.tolist()[0])
    print("shifted targets: ", targets.tolist()[0])
    print("training logits shape:", tuple(logits.shape))
    print("training loss (random weights, not a quality metric):", round(loss.item(), 4))
    print("causal future-invariance: PASS")
    print("toy softmax [2,1,0]:", [round(x, 4) for x in probabilities])
    print("generated toy IDs:", generated)
    print("autoregressive prefix length:", current.shape[1])


if __name__ == "__main__":
    main()
