# 🎙️ Whisper-Small Fine-Tuned on Uzbek Speech (1-Hour Dataset)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ZaaX19%2Fwhisper--small--uzbek--1h-blue.svg)](https://huggingface.co/ZaaX19/whisper-small-uzbek-1h)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Zizx4uM%2FWhisper--small-black.svg?logo=github)](https://github.com/Zizx4uM/Whisper-small)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/transformers/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Speech-to-Text (STT) model for the Uzbek language**, created by fine-tuning OpenAI's [`whisper-small`](https://huggingface.co/openai/whisper-small) (244M parameters) on the [`islomov/news_youtube_uzbek_speech_dataset`](https://huggingface.co/datasets/islomov/news_youtube_uzbek_speech_dataset) using Hugging Face Transformers and Google Colab T4 GPU.

---

## 📌 Project Overview

O'zbek tili uchun avtomatik nutqni tanish (Speech-to-Text / ASR) modelini yaratish va optimallashtirish. Ushbu loyiha quyidagi asosiy bosqichlarni o'z ichiga oladi:
1. **1 soatlik audioni ajratish:** Ko'p soatlik datasetdan aniq 3600 soniyalik audio qismini tanlab olish va 16 kHz chastotaga o'tkazish.
2. **Audio va Matn Preprocessing:** Whisper'ning 80 ta log-mel chastotali spektral xususiyatlarini (`WhisperFeatureExtractor`) hamda o'zbek tili lotin alifbosi tokenizatsiyasini (`WhisperTokenizer`) shakllantirish.
3. **Seq2Seq Fine-Tuning:** Colab T4 GPU'da `fp16` va `gradient_checkpointing` yordamida VRAM sarfini ~8GB darajasida saqlagan holda samarali o'qitish.
4. **Metrikalarni hisoblash:** Word Error Rate (WER) va Character Error Rate (CER) orqali model aniqligini baholash.
5. **Hugging Face Hub integratsiyasi:** Model og'irliklari, tokenizer va konfiguratsiyalarni to'g'ridan-to'g'ri [ZaaX19/whisper-small-uzbek-1h](https://huggingface.co/ZaaX19/whisper-small-uzbek-1h) reposiga yuklash.
6. **Interaktiv Web UI:** Gradio interfeysi orqali mikrofon yoki audio fayllarni real-vaqtda transkripsiya qilish.

---

## 📊 Training & Evaluation Results

Model Google Colab T4 GPU (16GB VRAM) da 600 qadam (steps) davomida to'liq Seq2Seq fine-tuning qilindi.

### Training Progression

| Step | Epoch | Training Loss | Validation Loss | WER (%) | CER (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **150** | 18.8 | 0.1019 | 1.0722 | 60.89% | 20.33% |
| **300** | 37.5 | 0.0043 | 1.0996 | 58.67% | 19.10% |
| **450** | 56.3 | 0.0016 | 1.1218 | **58.39%** | **18.96%** |
| **600** | 75.0 | 0.0013 | 1.1297 | 58.61% | 19.21% |

### Key Benchmark Takeaways
* **Best WER:** **58.38%** (at step 450)
* **Best CER (Character Error Rate):** **18.96%** (at step 450)
* **Tahlil:** Past Character Error Rate (~19%) model o'zbek tili fonetikasi va lotin alifbosidagi harflarni juda aniq o'zlashtirganini ko'rsatadi. WER ko'rsatkichidagi xatoliklarning katta qismi qo'shma so'zlar va tinish belgilarining transkripsiyasiga to'g'ri keladi.

---

## 📂 Repository Structure

```
.
├── LICENSE                                 # MIT License
├── README.md                               # Loyiha hujjatlari va natijalari
├── requirements.txt                        # Python kutubxonalari
├── app.py                                  # Gradio Web UI ilovasi
├── notebooks/
│   └── whisper_uzbek_finetune_colab.ipynb  # Colab 1-klik bilan ishga tushirish
└── src/
    ├── __init__.py
    ├── config.py                           # Giperparametrlar va sozlamalar
    ├── data_loader.py                      # 1 soatlik audioni ajratish va mel-spektrogramma
    ├── collator.py                         # Dinamik padding bilan Seq2Seq collator
    ├── metrics.py                          # WER va CER metrikalari
    ├── train.py                            # Asosiy CLI training skripti
    ├── evaluate_model.py                   # Baseline vs Fine-tuned taqqoslash skripti
    └── inference.py                        # CLI tezkor transkripsiya skripti
```

---

## 🎯 Quick Inference Example

Model o'qitilib Hugging Face Hub'ga joylangan. Uni istalgan muhitda 3 qator kod bilan ishlatish mumkin:

```python
from transformers import pipeline

# Hugging Face Hub'dan modelni yuklash
pipe = pipeline("automatic-speech-recognition", model="ZaaX19/whisper-small-uzbek-1h")

# O'zbekcha audio faylni matnga aylantirish
transcription = pipe("audio_namuna.mp3")
print("Olingan matn:", transcription["text"])
```

### CLI orqali ishlatish:
```bash
python3 -m src.inference --audio path/to/sample.mp3
```

---

## 🖥️ Interactive Web UI (Gradio)

Mikrofon orqali gapirib ko'rish yoki audio fayl yuklab natijani olish uchun:

```bash
python3 app.py
```
Brauzerda `http://localhost:7860` manzilini oching.

---

## 🚀 How to Run in Google Colab

Google Colab'da bepul T4 GPU bilan ishlash uchun:

1. [Google Colab](https://colab.research.google.com/) ni oching.
2. `Upload` bo'limi orqali [`notebooks/whisper_uzbek_finetune_colab.ipynb`](notebooks/whisper_uzbek_finetune_colab.ipynb) faylini yuklang.
3. **GPU yoqish:** `Runtime` -> `Change runtime type` -> **T4 GPU** tanlang.
4. **Hugging Face Token qo'shish:**
   - Chap paneldagi **🔑 Secrets** bo'limida `HF_TOKEN` nomli sir qo'shing.
5. Hujayralarni ketma-ket ishga tushiring (`Runtime` -> `Run all`).

---

## 💻 Local Setup & Execution

### 1. Muhitni o'rnatish
```bash
git clone https://github.com/Zizx4uM/Whisper-small.git
cd Whisper-small
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Modelni o'qitish
```bash
# Test / Dry-run rejimi (5 step)
python3 -m src.train --dry_run

# To'liq o'qitish va Hub'ga yuklash
python3 -m src.train \
    --model_id "openai/whisper-small" \
    --max_steps 600 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --hub_model_id "ZaaX19/whisper-small-uzbek-1h" \
    --push_to_hub
```

### 3. Baholash va Solishtirish (WER / CER)
```bash
python3 -m src.evaluate_model \
    --baseline_id "openai/whisper-small" \
    --finetuned_path "ZaaX19/whisper-small-uzbek-1h" \
    --num_samples 30
```

---

## ⚙️ Hyperparameters & Training Settings

| Parameter | Qiymati | Izoh |
| :--- | :--- | :--- |
| **Base Model** | `openai/whisper-small` | 244M parametrli Encoder-Decoder arxitekturasi |
| **Dataset** | `islomov/news_youtube_uzbek_speech_dataset` | 1 soatlik audio (~3600 soniya) |
| **Sampling Rate** | 16,000 Hz | Whisper uchun standart chastota |
| **Precision** | Native AMP (FP16) | T4 GPU xotirasini tejash va tezlashuv |
| **Gradient Checkpointing** | `True` | VRAM iste'molini 14GB dan ~7.5GB ga tushiradi |
| **Batch Size** | 8 (Grad Accum: 2) | Samarali batch size: 16 |
| **Optimizer** | AdamW (Torch Fused) | `betas=(0.9, 0.999)`, `eps=1e-08` |
| **Learning Rate** | 1e-5 | Linear scheduler, Warmup: 50 steps |
| **Steps** | 600 steps | ~75 epoch ga teng |

---

## 📜 Resume / Portfolio Highlights

* **End-to-End ASR Pipeline:** Audio extraction, resampling (16kHz), feature engineering (80-channel log-mel filterbanks) va Uzbek BPE tokenizatsiyasi.
* **Low-Resource Language Adaptation:** O'zbek tiliga xos lotin alifbosi va tovushlarni transfer learning orqali modelga muvaffaqiyatli moslashtirish (CER: 18.96%).
* **Hardware & VRAM Optimization:** Colab T4 (16GB VRAM) doirasida gradient checkpointing va FP16 yordamida VRAM iste'molini ~7.5 GB gacha qisqartirish.
* **Open Source Model Deployment:** Model, preprocessor va metric kartalarini Hugging Face Hub'ga avtomatik yuklash va Gradio Web UI yaratish.

---

## 📄 License
Ushbu loyiha [MIT](LICENSE) litsenziyasi asosida tarqatiladi.
