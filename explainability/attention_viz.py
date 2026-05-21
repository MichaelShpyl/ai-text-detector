"""
Explainability - Stage 9 of the pipeline (added in response to peer feedback
from Shane McDevitt on the original Padlet post).

Returns the mean attention weight per input token, taken from the final
attention layer of DistilBERT. This gives a lecturer a defensible answer to
the question 'why did the model say this is AI-generated?' during a viva.
"""
from typing import List, Tuple

import torch


def compute_attention(pipeline, text: str) -> Tuple[List[str], List[float]]:
    """Return (tokens, normalised_weights) for a single input."""
    tokenizer = pipeline.tokenizer
    model = pipeline.model

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions is a tuple of (layer_0, ..., layer_n);
    # each shape: (batch, n_heads, seq_len, seq_len). Use the final layer.
    last_layer = outputs.attentions[-1]  # (1, heads, seq, seq)
    # Average across heads, then take attention TO each token (column-wise mean)
    mean_attn = last_layer.mean(dim=1).squeeze(0).mean(dim=0)  # shape: (seq_len,)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
    weights = mean_attn.tolist()

    # Filter out special tokens for display
    visible = [(t, w) for t, w in zip(tokens, weights) if t not in ("[CLS]", "[SEP]", "[PAD]")]
    if not visible:
        return [], []

    out_tokens, out_weights = zip(*visible)
    # Normalise to sum to 1 for cleaner downstream display
    total = sum(out_weights) or 1.0
    out_weights = [w / total for w in out_weights]
    return list(out_tokens), out_weights
