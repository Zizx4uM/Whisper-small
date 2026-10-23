"""CLI and programmatic inference module for fine-tuned Whisper Uzbek model."""

import argparse
import sys
import time
from typing import Optional
import torch
from transformers import pipeline


def transcribe_audio(
    audio_path: str,
    model_id: str = "ZaaX19/whisper-small-uzbek",
    language: str = "uzbek",
    device: Optional[str] = None,
) -> str:
    """
    Transcribes an Uzbek audio file into text using the fine-tuned Whisper model.

    Args:
        audio_path: Path to the input audio file (WAV, MP3, FLAC, etc.).
        model_id: Hugging Face hub ID or local checkpoint path.
        language: Language spoken in audio.
        device: 'cuda', 'cpu', or None (auto-detected).

    Returns:
        The transcribed Uzbek text string.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        chunk_length_s=30,
    )

    result = pipe(
        audio_path,
        generate_kwargs={"language": language, "task": "transcribe"},
    )
    return result["text"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe Uzbek audio using fine-tuned Whisper-Small")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (mp3, wav, flac, etc.)")
    parser.add_argument(
        "--model_id",
        type=str,
        default="ZaaX19/whisper-small-uzbek",
        help="HF model ID or local checkpoint path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference ('cuda', 'cpu', or auto-detect)",
    )
    args = parser.parse_args()

    print(f"Loading model '{args.model_id}'...")
    start_time = time.time()
    transcription = transcribe_audio(args.audio, model_id=args.model_id, device=args.device)
    elapsed = time.time() - start_time

    print("\n" + "=" * 50)
    print("TRANSCRIPTION RESULT:")
    print("=" * 50)
    print(transcription)
    print("-" * 50)
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    main()
