"""Interactive Gradio Web UI for Whisper-Small Uzbek Speech-to-Text."""

import os
import time
import torch
import gradio as gr
from transformers import pipeline

MODEL_ID = os.getenv("HF_MODEL_ID", "ZaaX19/whisper-small-uzbek")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Loading Whisper Uzbek pipeline from '{MODEL_ID}' on device '{DEVICE}'...")
pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    device=DEVICE,
    chunk_length_s=30,
)


def transcribe(audio_path: str):
    """Transcribes input audio (from microphone or upload) to Uzbek text."""
    if not audio_path:
        return "Iltimos, audio yozing yoki fayl yuklang.", "0.00 s"

    start_time = time.time()
    result = pipe(
        audio_path,
        generate_kwargs={"language": "uzbek", "task": "transcribe"},
    )
    elapsed = time.time() - start_time
    return result["text"].strip(), f"{elapsed:.2f} s"


demo = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Ovozli xabar / Audio fayl (Microphone or Audio File)",
        )
    ],
    outputs=[
        gr.Textbox(label="O'zbekcha transkripsiya (Uzbek Transcription)", lines=4),
        gr.Label(label="Qayta ishlash vaqti (Processing Time)"),
    ],
    title="🎙️ Whisper-Small Uzbek Speech-to-Text",
    description=(
        "OpenAI `whisper-small` modelining o'zbek tili uchun moslashtirilgan (fine-tuned) talqini. "
        "Audioni mikrofon orqali yozib oling yoki audio fayl (WAV, MP3, FLAC) yuklang."
    ),
    article=(
        "### Loyiha haqida (About Project)\n"
        "- **Hugging Face Model:** [ZaaX19/whisper-small-uzbek](https://huggingface.co/ZaaX19/whisper-small-uzbek)\n"
        "- **GitHub Repository:** [Zizx4uM/Whisper-small](https://github.com/Zizx4uM/Whisper-small)\n"
        "- **Dataset:** `islomov/news_youtube_uzbek_speech_dataset`\n"
        "- **Metrikalar (Benchmark):** CER: **18.96%** | WER: **58.38%**\n"
    ),
    theme="soft",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
