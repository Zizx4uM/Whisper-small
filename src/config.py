"""Configuration module for Whisper fine-tuning on Uzbek speech."""

from dataclasses import dataclass, field
import os
from typing import Optional


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration settings for Whisper fine-tuning."""

    # Model & Tokenizer
    model_id: str = "openai/whisper-small"
    language: str = "uzbek"
    language_code: str = "uz"
    task: str = "transcribe"

    # Dataset & Preprocessing
    dataset_name: str = "islomov/news_youtube_uzbek_speech_dataset"
    dataset_config: Optional[str] = None
    target_duration_seconds: Optional[float] = None  # None for full dataset or float seconds for subset experiments
    sampling_rate: int = 16000
    test_size: float = 0.2
    random_seed: int = 42

    # Hardware & Training
    output_dir: str = "./whisper-small-uzbek"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    warmup_steps: int = 50
    max_steps: int = 600
    gradient_checkpointing: bool = True
    fp16: bool = True
    eval_steps: int = 150
    save_steps: int = 150
    logging_steps: int = 25
    save_total_limit: int = 2

    # Generation & Evaluation
    predict_with_generate: bool = True
    generation_max_length: int = 225
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False

    # Hugging Face Hub Integration
    push_to_hub: bool = True
    hub_model_id: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_HUB_MODEL_ID", "ZaaX19/whisper-small-uzbek")
    )
    hf_token: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_TOKEN", None)
    )
