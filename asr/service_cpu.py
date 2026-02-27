"""
ASR-микросервис на базе faster-whisper (CPU).
Принимает аудиофайл (любой формат), приводит к WAV 16 кГц моно,
транскрибирует через Whisper и возвращает сегменты с временными метками.

Health-эндпоинт доступен сразу при старте:
  {"status": "loading"} — модель ещё загружается
  {"status": "ok"}      — готов к работе
"""

import os
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel
from pydantic import BaseModel

MODEL_SIZE   = os.getenv("ASR_MODEL_SIZE", "small")
LANGUAGE     = os.getenv("ASR_LANGUAGE", "ru")
COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")
BEAM_SIZE    = int(os.getenv("ASR_BEAM_SIZE", "5"))
CPU_THREADS  = int(os.getenv("ASR_CPU_THREADS", "4"))

# Глобальное состояние модели
_model: Optional[WhisperModel] = None
_model_ready = threading.Event()   # устанавливается когда модель загружена
_model_error: Optional[str] = None


def _load_model():
    """Загружает модель в фоновом потоке."""
    global _model, _model_error
    try:
        print(f"[ASR] Loading whisper/{MODEL_SIZE} on CPU "
              f"(compute={COMPUTE_TYPE}, threads={CPU_THREADS})…")
        _model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type=COMPUTE_TYPE,
            cpu_threads=CPU_THREADS,
        )
        print("[ASR] Model ready.")
    except Exception as exc:
        _model_error = str(exc)
        print(f"[ASR] Model load failed: {exc}")
    finally:
        _model_ready.set()   # разблокируем в любом случае (чтобы /transcribe мог вернуть 503)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Запускаем загрузку модели в фоне — uvicorn поднимается немедленно
    t = threading.Thread(target=_load_model, daemon=True)
    t.start()
    yield


app = FastAPI(title="Whisper ASR Service (CPU)", lifespan=lifespan)


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
    Возвращает статус сервиса:
      loading — модель ещё грузится
      error   — загрузка завершилась ошибкой
      ok      — готов принимать запросы
    """
    if not _model_ready.is_set():
        return {"status": "loading", "model": MODEL_SIZE, "device": "cpu"}
    if _model_error:
        return {"status": "error", "model": MODEL_SIZE, "device": "cpu", "detail": _model_error}
    return {"status": "ok", "model": MODEL_SIZE, "device": "cpu"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...), task_id: str = ""):
    if not _model_ready.is_set():
        raise HTTPException(status_code=503, detail="Model is still loading, retry later")
    if _model_error:
        raise HTTPException(status_code=503, detail=f"Model failed to load: {_model_error}")

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments_iter, info = _model.transcribe(
            tmp_path,
            language=LANGUAGE if LANGUAGE != "auto" else None,
            beam_size=BEAM_SIZE,
        )
        segments: List[Segment] = [
            Segment(start=s.start, end=s.end, text=s.text.strip())
            for s in segments_iter
            if s.text.strip()
        ]
        return TranscribeResponse(
            task_id=task_id or None,
            language=info.language,
            full_text=" ".join(s.text for s in segments),
            segments=segments,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)
