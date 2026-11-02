"""Completion-only next-token loss; exclude all padding (MLX-LM 0.32.2 boundary fix)."""
import mlx.core as mx
import mlx.nn as nn

def completion_loss(model, batch, lengths):
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs)
    # Target positions refer to original full sequence: 1 .. padded_length-1.
    positions = mx.arange(1, targets.shape[1] + 1)
    mask = (positions >= lengths[:, :1]) & (positions < lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets)
    ntokens = mask.sum()
    return (ce * mask).astype(mx.float32).sum() / ntokens, ntokens
