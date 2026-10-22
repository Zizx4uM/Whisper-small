# 🎙️ Whisper-Small Fine-Tuned on Uzbek Speech (1-Hour Dataset)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/transformers/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Speech-to-Text (STT) model for the Uzbek language**, created by fine-tuning OpenAI's [`whisper-small`](https://huggingface.co/openai/whisper-small) on the [`islomov/news_youtube_uzbek_speech_dataset`](https://huggingface.co/datasets/islomov/news_youtube_uzbek_speech_dataset) using Hugging Face Transformers and Google Colab T4 GPU.

---

## 📌 Project Overview

O'zbek tili uchun avtomatik nutqni tanish (Speech-to-Text / ASR) modelini yaratish va optimallashtirish. Ushbu loyiha quyidagi asosiy bosqichlarni o'z ichiga oladi:
1. **1 soatlik audioni ajratish:** Ko'p soatlik datasetdan aniq 3600 soniyalik audio qismini tanlab olish va 16 kHz chastotaga o'tkazish.
2. **Audio va Matn Preprocessing:** Whisper'ning 80 ta log-mel chastotali spektral xususiyatlarini (`WhisperFeatureExtractor`) hamda o'zbek tili lotin alifbosi tokenizatsiyasini (`WhisperTokenizer`) shakllantirish.
3. **Seq2Seq Fine-Tuning:** Colab T4 GPU'da `fp16` va `gradient_checkpointing` yordamida VRAM sarfini ~8GB darajasida saqlagan holda samarali o'qitish.
4. **Metrikalarni hisoblash:** Word Error Rate (WER) va Character Error Rate (CER) orqali model aniqligini baholash.
5. **Hugging Face Hub integratsiyasi:** Model og'irliklari, tokenizer va konfiguratsiyalarni to'g'ridan-to'g'ri Hugging Face Hub'ga joylash.

---

## 📂 Repository Structure

```
/home/zaax/Projects/whisper/
├── .gitignore
├── requirements.txt
├── README.md
├── notebooks/
│   └── whisper_uzbek_finetune_colab.ipynb  # Colab 1-klik bilan ishga tushirish
└── src/
    ├── __init__.py
    ├── config.py                           # Giperparametrlar va sozlamalar
    ├── data_loader.py                      # 1 soatlik audioni ajratish va mel-spektrogramma
    ├── collator.py                         # Dinamik padding bilan Seq2Seq collator
    ├── metrics.py                          # WER va CER metrikalari
    ├── train.py                            # Asosiy CLI training skripti
    └── evaluate_model.py                   # Baseline vs Fine-tuned taqqoslash skripti
```

---

## 🚀 How to Run in Google Colab (Recommended)

Google Colab'da bepul T4 GPU bilan ishlash uchun:

1. [Google Colab](https://colab.research.google.com/) ni oching.
2. `Upload` bo'limi orqali [`notebooks/whisper_uzbek_finetune_colab.ipynb`](file:///home/zaax/Projects/whisper/notebooks/whisper_uzbek_finetune_colab.ipynb) faylini yuklang.
3. **GPU yoqish:** `Runtime` -> `Change runtime type` -> **T4 GPU** tanlang.
4. **Hugging Face Token qo'shish:**
   - Chap paneldagi **🔑 Secrets** tugmasini bosing.
   - Yangi sir yarating: Name: `HF_TOKEN`, Value: sizning Hugging Face write tokeningiz.
   - **Notebook access** ni yoqing.
5. Hujayralarni ketma-ket ishga tushiring (`Runtime` -> `Run all`).

---

## 💻 Local Setup & Execution

### 1. Muhitni o'rnatish
```bash
git clone <your-repo-url>
cd whisper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Hugging Face tizimiga kirish
```bash
export HF_TOKEN="your_hf_write_token_here"
# yoki
huggingface-cli login
```

### 3. Modelni o'qitish
```bash
# Test / Dry-run rejimi (5 step)
python3 -m src.train --dry_run

# To'liq o'qitish va Hub'ga yuklash
python3 -m src.train \
    --model_id "openai/whisper-small" \
    --max_steps 600 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --hub_model_id "YOUR_USERNAME/whisper-small-uzbek-1h" \
    --push_to_hub
```

### 4. Baholash va Solishtirish (WER / CER)
```bash
python3 -m src.evaluate_model \
    --baseline_id "openai/whisper-small" \
    --finetuned_path "./whisper-small-uzbek-1h" \
    --num_samples 30
```

---

## 📊 Training Parameters & Architecture

| Parameter | Qiymati | Izoh |
| :--- | :--- | :--- |
| **Base Model** | `openai/whisper-small` | 244M parametrli Encoder-Decoder arxitekturasi |
| **Dataset** | `islomov/news_youtube_uzbek_speech_dataset` | 1 soatlik audio (~3600 soniya) |
| **Sampling Rate** | 16,000 Hz | Whisper uchun standart chastota |
| **Precision** | FP16 | T4 GPU xotirasini tejash va tezlashuv |
| **Gradient Checkpointing** | `True` | VRAM iste'molini 14GB dan ~7.5GB ga tushiradi |
| **Batch Size** | 8 (Grad Accum: 2) | Samarali batch size: 16 |
| **Learning Rate** | 1e-5 | AdamW optimizatori, Warmup: 50 steps |
| **Steps / Epochs** | 600 steps | ~3.5 epoch ga to'g'ri keladi |

---

## 🎯 Quick Inference Example

Model o'qitilib Hugging Face Hub'ga yuklangandan so'ng, uni istalgan joyda 3 qator kod bilan ishlatish mumkin:

```python
from transformers import pipeline

# Hugging Face'dan modelni yuklash
pipe = pipeline("automatic-speech-recognition", model="YOUR_USERNAME/whisper-small-uzbek-1h")

# O'zbekcha audio faylni matnga aylantirish
transcription = pipe("audio_namuna.mp3")
print("Olingan matn:", transcription["text"])
```

---

## 📜 Resume / Portfolio Highlights

* **End-to-End ASR Pipeline:** Audio extraction, resampling (16kHz), feature engineering (80-channel log-mel filterbanks) va Uzbek BPE tokenizatsiyasi.
* **Low-Resource Language Adaptation:** O'zbek tiliga xos lotin alifbosi va tovushlarni transfer learning orqali modelga muvaffaqiyatli moslashtirish.
* **Hardware & VRAM Optimization:** Colab bepul T4 (16GB VRAM) doirasida gradient checkpointing va FP16 yordamida OOM xatolarisiz modelni to'liq fine-tune qilish.
* **Open Source Model Deployment:** Model, preprocessor va metric kartalarini Hugging Face Hub'ga avtomatik integratsiya qilish.

---

## 📄 Litsenziya
Ushbu loyiha [MIT](LICENSE) litsenziyasi asosida tarqatiladi.
