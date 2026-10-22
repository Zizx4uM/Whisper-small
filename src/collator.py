"""Speech sequence-to-sequence data collator with dynamic padding."""

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that dynamically pads input audio features and target text tokens.
    Padding tokens in the labels are replaced with -100 to ensure they are ignored
    by the cross-entropy loss function.
    """

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Extract and pad input features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Extract and pad labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore during loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended at the start of labels, strip it
        bos_id = self.processor.tokenizer.bos_token_id
        if bos_id is not None and labels.shape[1] > 0 and (labels[:, 0] == bos_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
