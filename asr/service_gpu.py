"""
ASR-микросервис на базе Qwen3-ASR.
Принимает WAV (любой частоты / каналов), приводит к 16 кГц моно,
делит на чанки и возвращает транскрипцию с временными метками.
"""

import os
import tempfile
import threading
from typing import List, Optional

import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from qwen_asr import Qwen3ASRModel

MODEL_ID       = os.getenv("ASR_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
LANGUAGE       = os.getenv("ASR_LANGUAGE", "Russian")   # "auto" → None
CHUNK_S        = int(os.getenv("ASR_CHUNK_S", "30"))
MAX_NEW_TOKENS = int(os.getenv("ASR_MAX_TOKENS", "256"))
SILENCE_THRESH = float(os.getenv("ASR_SILENCE_THRESH", "0.01"))
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

_model: Optional[Qwen3ASRModel] = None
_model_ready = threading.Event()
_model_error: Optional[str] = None


def _load_model():
    global _model, _model_error
    try:
        print(f"[ASR] Loading {MODEL_ID} on {DEVICE}…")
        _model = Qwen3ASRModel.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            device_map=DEVICE,
            max_inference_batch_size=8,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        print("[ASR] Ready.")
    except Exception as e:
        _model_error = str(e)
        print(f"[ASR] Failed to load model: {e}")
    finally:
        _model_ready.set()


# Запускаем загрузку модели в фоне — FastAPI стартует немедленно
threading.Thread(target=_load_model, daemon=True).start()

app = FastAPI(title="Qwen3-ASR Service")


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    task_id: Optional[str]
    language: Optional[str]
    full_text: str
    segments: List[Segment]


@app.get("/health")
async def health():
    """
    Отвечает немедленно:
    - status=loading  пока модель грузится
    - status=ready    когда модель готова к работе
    - status=error    если загрузка провалилась
    """
    if _model_error:
        return {"status": "error", "model": MODEL_ID, "device": DEVICE, "detail": _model_error}
    if not _model_ready.is_set():
        return {"status": "loading", "model": MODEL_ID, "device": DEVICE}
    return {"status": "ready", "model": MODEL_ID, "device": DEVICE}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...), task_id: str = ""):
    if not _model_ready.is_set():
        raise HTTPException(status_code=503, detail="Model is still loading, please retry later")
    if _model_error:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {_model_error}")

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        waveform, sr = torchaudio.load(tmp_path)

        # моно
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 16 кГц
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            sr = 16000

        total = waveform.shape[1]
        chunk = CHUNK_S * sr
        segments: List[Segment] = []
        detected_lang: Optional[str] = None

        for start in range(0, total, chunk):
            end   = min(start + chunk, total)
            piece = waveform[:, start:end]

            if torch.max(torch.abs(piece)).item() < SILENCE_THRESH:
                continue

            results = _model.transcribe(
                audio=(piece.squeeze().numpy(), sr),
                language=LANGUAGE if LANGUAGE != "auto" else None,
            )
            text = (results[0].text or "").strip()
            if detected_lang is None:
                detected_lang = getattr(results[0], "language", None)
            if text:
                segments.append(Segment(start=start / sr, end=end / sr, text=text))

        return TranscribeResponse(
            task_id=task_id or None,
            language=detected_lang,
            full_text=" ".join(s.text for s in segments),
            segments=segments,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)
