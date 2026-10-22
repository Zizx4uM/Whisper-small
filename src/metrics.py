"""Evaluation metrics (WER and CER) calculation for speech recognition."""

from typing import Callable, Dict, Any
import numpy as np
import evaluate
from transformers import WhisperTokenizer


def build_compute_metrics_fn(tokenizer: WhisperTokenizer) -> Callable[[Any], Dict[str, float]]:
    """
    Constructs a compute_metrics function compatible with Hugging Face Seq2SeqTrainer.

    Args:
        tokenizer: Initialized WhisperTokenizer for Uzbek decoding.

    Returns:
        Callable returning dictionary with 'wer' and 'cer' error percentages.
    """
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred: Any) -> Dict[str, float]:
        pred_ids = pred.predictions
        # Safe replacement without mutating original array in-place
        label_ids = np.where(pred.label_ids != -100, pred.label_ids, tokenizer.pad_token_id)

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Filter out empty reference strings to prevent division by zero in jiwer
        valid_pairs = [(p, r) for p, r in zip(pred_str, label_str) if len(r.strip()) > 0]
        if not valid_pairs:
            return {"wer": 0.0, "cer": 0.0}

        filtered_preds, filtered_refs = zip(*valid_pairs)
        wer = 100.0 * wer_metric.compute(predictions=list(filtered_preds), references=list(filtered_refs))
        cer = 100.0 * cer_metric.compute(predictions=list(filtered_preds), references=list(filtered_refs))

        return {"wer": wer, "cer": cer}

    return compute_metrics
