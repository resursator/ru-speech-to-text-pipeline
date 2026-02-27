"""
ASR-микросервис на базе faster-whisper (CPU).
Принимает аудиофайл (любой формат), приводит к WAV 16 кГц моно,
транскрибирует через Whisper и возвращает сегменты с временными метками.
"""

import os
import tempfile
import threading
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel
from pydantic import BaseModel

MODEL_SIZE     = os.getenv("ASR_MODEL_SIZE", "small")   # tiny, base, small, medium, large-v3
LANGUAGE       = os.getenv("ASR_LANGUAGE", "ru")        # None = авто-определение
COMPUTE_TYPE   = os.getenv("ASR_COMPUTE_TYPE", "int8")  # int8 быстрее на CPU
BEAM_SIZE      = int(os.getenv("ASR_BEAM_SIZE", "5"))
CPU_THREADS    = int(os.getenv("ASR_CPU_THREADS", "4"))

_model: Optional[WhisperModel] = None
_model_ready = threading.Event()
_model_error: Optional[str] = None


def _load_model():
    global _model, _model_error
    try:
        print(f"[ASR] Loading whisper/{MODEL_SIZE} on CPU (compute={COMPUTE_TYPE}, threads={CPU_THREADS})…")
        _model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type=COMPUTE_TYPE,
            cpu_threads=CPU_THREADS,
        )
        print("[ASR] Ready.")
    except Exception as e:
        _model_error = str(e)
        print(f"[ASR] Failed to load model: {e}")
    finally:
        _model_ready.set()


# Запускаем загрузку модели в фоне — FastAPI стартует немедленно
threading.Thread(target=_load_model, daemon=True).start()

app = FastAPI(title="Whisper ASR Service (CPU)")


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
        return {"status": "error", "model": MODEL_SIZE, "device": "cpu", "detail": _model_error}
    if not _model_ready.is_set():
        return {"status": "loading", "model": MODEL_SIZE, "device": "cpu"}
    return {"status": "ready", "model": MODEL_SIZE, "device": "cpu"}


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
