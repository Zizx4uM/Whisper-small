"""Main training script for fine-tuning Whisper on Uzbek speech."""

import argparse
import logging
import os
import sys
from huggingface_hub import login
from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from src.config import TrainingConfig
from src.data_loader import load_and_slice_dataset
from src.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.metrics import build_compute_metrics_fn

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Uzbek Speech Dataset")
    parser.add_argument("--model_id", type=str, default="openai/whisper-small", help="Hugging Face model ID")
    parser.add_argument("--max_steps", type=int, default=600, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Per device train batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./whisper-small-uzbek", help="Output directory")
    parser.add_argument("--hub_model_id", type=str, default="ZaaX19/whisper-small-uzbek", help="Target Hugging Face Hub model ID")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and processor to Hugging Face Hub")
    parser.add_argument("--dry_run", action="store_true", help="Run 5 steps dry-run test mode")
    args = parser.parse_args()

    config = TrainingConfig()
    config.model_id = args.model_id
    config.max_steps = 5 if args.dry_run else args.max_steps
    config.per_device_train_batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.output_dir = args.output_dir
    if args.hub_model_id:
        config.hub_model_id = args.hub_model_id
        config.push_to_hub = True
    elif args.push_to_hub:
        config.push_to_hub = True

    # Check Hugging Face authentication
    hf_token = os.getenv("HF_TOKEN") or config.hf_token
    if hf_token:
        logger.info("Logging into Hugging Face Hub with environment token...")
        login(token=hf_token)
    elif config.push_to_hub:
        logger.warning(
            "push_to_hub is True but HF_TOKEN environment variable is not set. "
            "Ensure you are logged in via `huggingface-cli login` or set HF_TOKEN."
        )

    # 1. Load dataset & processor
    logger.info("Loading and preprocessing dataset...")
    dataset, processor = load_and_slice_dataset(config)

    # 2. Initialize model
    logger.info("Loading pretrained model %s...", config.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(config.model_id)

    # Disable forced decoder IDs and token suppression to allow free generation in Uzbek
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    # Gradient checkpointing requires cache to be disabled
    model.config.use_cache = False

    # Explicitly configure generation settings for Uzbek STT evaluation
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.language = config.language
        model.generation_config.task = config.task
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []

    # 3. Data collator & metrics
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics_fn = build_compute_metrics_fn(processor.tokenizer)

    # 4. Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps if not args.dry_run else 1,
        max_steps=config.max_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=config.fp16,
        eval_strategy="steps",
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        predict_with_generate=config.predict_with_generate,
        generation_max_length=config.generation_max_length,
        save_steps=config.save_steps if not args.dry_run else 5,
        eval_steps=config.eval_steps if not args.dry_run else 5,
        logging_steps=config.logging_steps if not args.dry_run else 1,
        save_total_limit=config.save_total_limit,
        report_to=["tensorboard"],
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        push_to_hub=config.push_to_hub and (config.hub_model_id is not None),
        hub_model_id=config.hub_model_id,
        hub_token=hf_token,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=processor.feature_extractor,
    )

    # 5. Train
    logger.info("Starting training loop (Total steps: %d)...", config.max_steps)
    train_result = trainer.train()
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 6. Evaluation
    logger.info("Running evaluation on test split...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("Evaluation results: WER=%.2f%%, CER=%.2f%%",
                eval_metrics.get("eval_wer", 0.0), eval_metrics.get("eval_cer", 0.0))

    # 7. Push to Hub if configured
    if config.push_to_hub and config.hub_model_id:
        logger.info("Pushing model, tokenizer, and processor to Hugging Face Hub (%s)...",
                    config.hub_model_id)
        trainer.push_to_hub()
        processor.push_to_hub(config.hub_model_id, token=hf_token)
        logger.info("Successfully pushed model to https://huggingface.co/%s", config.hub_model_id)

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
