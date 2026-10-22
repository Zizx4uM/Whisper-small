"""Evaluation and benchmark script comparing baseline Whisper with the fine-tuned model."""

import argparse
import logging
from typing import Dict, Any, List
import torch
import evaluate
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    model_id_or_path: str,
    test_samples: List[Dict[str, Any]],
    language: str = "uzbek",
    task: str = "transcribe",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """
    Evaluates a Whisper checkpoint on a given set of audio test samples.

    Args:
        model_id_or_path: Hugging Face hub ID or local directory path.
        test_samples: List of dicts with 'audio' and 'text'.
        language: Language for transcription.
        task: Whisper task ('transcribe').
        device: Device to run inference on.

    Returns:
        Dictionary with 'wer' and 'cer' error percentages.
    """
    logger.info("Loading model and processor from %s on device %s...", model_id_or_path, device)
    processor = WhisperProcessor.from_pretrained(model_id_or_path, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_id_or_path).to(device)
    model.eval()

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    predictions = []
    references = []

    for i, sample in enumerate(test_samples):
        audio_array = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]

        input_features = processor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language=language,
                task=task,
                max_new_tokens=225,
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        reference = sample["text"]

        predictions.append(transcription)
        references.append(reference)

        if i < 3:
            logger.info("Sample %d Reference:  %s", i + 1, reference)
            logger.info("Sample %d Prediction: %s", i + 1, transcription)

    wer = 100.0 * wer_metric.compute(predictions=predictions, references=references)
    cer = 100.0 * cer_metric.compute(predictions=predictions, references=references)

    return {"wer": wer, "cer": cer}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs fine-tuned Whisper model on Uzbek")
    parser.add_argument("--finetuned_path", type=str, required=True, help="Local path or HF hub ID of fine-tuned model")
    parser.add_argument("--baseline_id", type=str, default="openai/whisper-small", help="Baseline Whisper model ID")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of test samples to evaluate on")
    args = parser.parse_args()

    logger.info("Loading test audio samples from Uzbek dataset...")
    dataset = load_dataset("islomov/news_youtube_uzbek_speech_dataset", split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Take the last `num_samples` to avoid train split contamination
    total_len = len(dataset)
    test_subset = dataset.select(range(max(0, total_len - args.num_samples), total_len))
    samples = [sample for sample in test_subset]

    logger.info("=== 1. Evaluating Baseline Model (%s) ===", args.baseline_id)
    baseline_scores = evaluate_checkpoint(args.baseline_id, samples)

    logger.info("=== 2. Evaluating Fine-Tuned Model (%s) ===", args.finetuned_path)
    finetuned_scores = evaluate_checkpoint(args.finetuned_path, samples)

    print("\n" + "=" * 60)
    print("           BENCHMARK COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Model':<30} | {'WER (%)':<12} | {'CER (%)':<12}")
    print("-" * 60)
    print(f"{args.baseline_id:<30} | {baseline_scores['wer']:<12.2f} | {baseline_scores['cer']:<12.2f}")
    print(f"{args.finetuned_path:<30} | {finetuned_scores['wer']:<12.2f} | {finetuned_scores['cer']:<12.2f}")
    print("=" * 60)

    wer_diff = baseline_scores["wer"] - finetuned_scores["wer"]
    cer_diff = baseline_scores["cer"] - finetuned_scores["cer"]
    print(f"Absolute WER Reduction: {wer_diff:+.2f}%")
    print(f"Absolute CER Reduction: {cer_diff:+.2f}%")


if __name__ == "__main__":
    main()
