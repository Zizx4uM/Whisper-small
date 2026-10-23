"""Unit tests for Whisper Uzbek STT training configuration."""

import unittest
from src.config import TrainingConfig


class TestTrainingConfig(unittest.TestCase):
    """Test suite for TrainingConfig dataclass."""

    def test_default_config_initialization(self):
        """Verify that default config parameters match expected settings."""
        config = TrainingConfig()
        self.assertEqual(config.model_id, "openai/whisper-small")
        self.assertEqual(config.language, "uzbek")
        self.assertEqual(config.language_code, "uz")
        self.assertEqual(config.task, "transcribe")
        self.assertEqual(config.sampling_rate, 16000)
        self.assertEqual(config.output_dir, "./whisper-small-uzbek")
        self.assertEqual(config.hub_model_id, "ZaaX19/whisper-small-uzbek")
        self.assertEqual(config.metric_for_best_model, "wer")
        self.assertFalse(config.greater_is_better)

    def test_custom_config_override(self):
        """Verify custom parameter overrides."""
        config = TrainingConfig(
            max_steps=100,
            per_device_train_batch_size=4,
            learning_rate=2e-5,
            target_duration_seconds=1800.0,
        )
        self.assertEqual(config.max_steps, 100)
        self.assertEqual(config.per_device_train_batch_size, 4)
        self.assertEqual(config.learning_rate, 2e-5)
        self.assertEqual(config.target_duration_seconds, 1800.0)


if __name__ == "__main__":
    unittest.main()
