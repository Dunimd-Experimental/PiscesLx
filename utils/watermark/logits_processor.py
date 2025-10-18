from typing import Optional, List
import torch

# HuggingFace-style logits processor to inject watermark bias during generation
# This integrates the lexical watermarking at sampling time.

class PiscesWatermarkLogitsProcessor:
    """
    A lightweight logits processor that boosts signature-bucket tokens.
    Compatible with HuggingFace `logits_processor` parameter of `generate()`.
    """
    def __init__(self, vocab_size: Optional[int] = None, seed: int = 12345, boost: float = 0.15):
        self.vocab_size = vocab_size
        self.seed = int(seed)
        self.boost = float(boost)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: [batch, vocab] or [vocab]
        from utils.watermark.text_lexical import apply_watermark_logits
        if scores.dim() == 1:
            vsize = self.vocab_size or scores.shape[0]
            return apply_watermark_logits(scores, vsize, self.seed, self.boost)
        elif scores.dim() == 2:
            vsize = self.vocab_size or scores.shape[-1]
            adjusted = []
            for row in scores:
                adjusted.append(apply_watermark_logits(row, vsize, self.seed, self.boost))
            return torch.stack(adjusted, dim=0)
        else:
            return scores