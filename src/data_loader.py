"""Dataset loading, audio preprocessing, and feature extraction for Uzbek speech recognition."""

import logging
from typing import Tuple, Dict, Any
from datasets import load_dataset, Audio, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

from src.config import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_slice_dataset(config: TrainingConfig) -> Tuple[DatasetDict, WhisperProcessor]:
    """
    Loads Uzbek speech dataset, prepares audio samples, and extracts Whisper features.

    Args:
        config: TrainingConfig object containing hyperparameters and paths.

    Returns:
        Tuple containing the prepared DatasetDict ('train' and 'test') and WhisperProcessor.
    """
    logger.info("Initializing Whisper feature extractor, tokenizer, and processor for %s (%s)...",
                config.model_id, config.language)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model_id)
    tokenizer = WhisperTokenizer.from_pretrained(
        config.model_id, language=config.language, task=config.task
    )
    processor = WhisperProcessor.from_pretrained(
        config.model_id, language=config.language, task=config.task
    )

    logger.info("Loading dataset %s...", config.dataset_name)
    raw_dataset = load_dataset(config.dataset_name, split="train")

    logger.info("Resampling audio to %d Hz...", config.sampling_rate)
    raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=config.sampling_rate))

    # Slice subset if target_duration_seconds is specified, otherwise use entire dataset
    if config.target_duration_seconds is not None and config.target_duration_seconds > 0:
        logger.info("Selecting audio samples up to %.1f seconds...", config.target_duration_seconds)
        selected_indices = []
        total_duration_sec = 0.0

        for idx, sample in enumerate(raw_dataset):
            audio_array = sample["audio"]["array"]
            duration = len(audio_array) / config.sampling_rate
            if total_duration_sec + duration > config.target_duration_seconds:
                break
            total_duration_sec += duration
            selected_indices.append(idx)

        logger.info("Selected %d samples totaling %.2f minutes (%.2f seconds).",
                    len(selected_indices), total_duration_sec / 60.0, total_duration_sec)
        subset = raw_dataset.select(selected_indices)
    else:
        logger.info("Using full dataset (%d samples).", len(raw_dataset))
        subset = raw_dataset

    # Train / Test split
    split_datasets = subset.train_test_split(
        test_size=config.test_size, seed=config.random_seed
    )
    logger.info("Split dataset into %d train samples and %d test samples.",
                len(split_datasets["train"]), len(split_datasets["test"]))

    # Prepare features
    def prepare_sample(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Tokenize target text
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch

    logger.info("Transforming audio into log-mel spectrograms and tokenizing text...")
    column_names = split_datasets["train"].column_names
    prepared_dataset = split_datasets.map(
        prepare_sample,
        remove_columns=column_names,
        num_proc=1,
        desc="Preprocessing audio & text"
    )

    return prepared_dataset, processor
