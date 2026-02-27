"""
ASR-сервис на базе официального пакета qwen-asr.
Принимает WAV (16кГц, моно), делит на 30-секундные сегменты,
возвращает список сегментов с временными метками.
"""

import os
import tempfile
import torch
import torchaudio
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from qwen_asr import Qwen3ASRModel

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
MODEL_ID      = os.getenv("ASR_MODEL_ID",       "Qwen/Qwen3-ASR-1.7B")
LANGUAGE      = os.getenv("ASR_LANGUAGE",        "Russian")   # None = авто-определение
CHUNK_LENGTH_S = int(os.getenv("ASR_CHUNK_S",   "30"))
MAX_NEW_TOKENS = int(os.getenv("ASR_MAX_TOKENS", "256"))
SILENCE_THRESH = float(os.getenv("ASR_SILENCE",  "0.01"))
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Модель — загружается один раз при старте
# ---------------------------------------------------------------------------
_model: Optional[Qwen3ASRModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    print(f"[ASR] Loading {MODEL_ID} on {DEVICE}...")
    _model = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
        max_inference_batch_size=8,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print("[ASR] Model ready.")
    yield


app = FastAPI(title="Qwen3-ASR Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Схемы ответа
# ---------------------------------------------------------------------------
class Segment(BaseModel):
    start: float
    end:   float
    text:  str


class TranscribeResponse(BaseModel):
    task_id:     Optional[str]
    language:    Optional[str]
    full_text:   str
    segments:    List[Segment]


# ---------------------------------------------------------------------------
# Эндпоинт
# ---------------------------------------------------------------------------
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file:    UploadFile = File(...),
    task_id: str        = "",
):
    """
    Принимает WAV-файл (16кГц, моно).
    Нарезает на CHUNK_LENGTH_S-секундные куски, транскрибирует каждый
    через qwen-asr, возвращает сегменты + склеенный полный текст.
    """
    # --- сохраняем загруженный файл во временный WAV ---
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # --- загружаем аудио ---
        waveform, sr = torchaudio.load(tmp_path)

        # приводим к моно
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # ресемплируем до 16 кГц
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            sr = 16000

        total_samples  = waveform.shape[1]
        chunk_samples  = CHUNK_LENGTH_S * sr
        segments: List[Segment] = []
        detected_lang: Optional[str] = None

        # --- обрабатываем по сегментам ---
        for start in range(0, total_samples, chunk_samples):
            end   = min(start + chunk_samples, total_samples)
            chunk = waveform[:, start:end]   # (1, N)

            # пропускаем тишину
            if torch.max(torch.abs(chunk)).item() < SILENCE_THRESH:
                continue

            audio_np = chunk.squeeze().numpy()   # (N,)

            # qwen-asr принимает (np.ndarray, sr) tuple напрямую
            results = _model.transcribe(
                audio=(audio_np, sr),
                language=LANGUAGE if LANGUAGE != "auto" else None,
            )

            res = results[0]
            text = (res.text or "").strip()

            if detected_lang is None:
                detected_lang = getattr(res, "language", None)

            if text:
                segments.append(Segment(
                    start=start / sr,
                    end=end   / sr,
                    text=text,
                ))

        full_text = " ".join(s.text for s in segments)

        return TranscribeResponse(
            task_id=task_id or None,
            language=detected_lang,
            full_text=full_text,
            segments=segments,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}
