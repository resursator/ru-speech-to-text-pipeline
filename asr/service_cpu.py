"""
ASR-микросервис на базе faster-whisper (CPU).
Принимает аудиофайл (любой формат), приводит к WAV 16 кГц моно,
транскрибирует через Whisper и возвращает сегменты с временными метками.
"""

import asyncio
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel
from pydantic import BaseModel

MODEL_SIZE   = os.getenv("ASR_MODEL_SIZE", "small")
LANGUAGE     = os.getenv("ASR_LANGUAGE", "ru")
COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")
BEAM_SIZE    = int(os.getenv("ASR_BEAM_SIZE", "5"))
CPU_THREADS  = int(os.getenv("ASR_CPU_THREADS", "4"))

_model: Optional[WhisperModel] = None
_model_ready = threading.Event()
_model_error: Optional[str] = None

# Однопоточный executor — faster-whisper не thread-safe при параллельных вызовах
_executor = ThreadPoolExecutor(max_workers=1)


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


def _do_transcribe(tmp_path: str) -> tuple:
    """Синхронная транскрипция — выполняется в отдельном потоке, не блокируя event loop."""
    segments_iter, info = _model.transcribe(
        tmp_path,
        language=LANGUAGE if LANGUAGE != "auto" else None,
        beam_size=BEAM_SIZE,
    )
    segments = [
        Segment(start=s.start, end=s.end, text=s.text.strip())
        for s in segments_iter
        if s.text.strip()
    ]
    return segments, info.language


@app.get("/health")
async def health():
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
        loop = asyncio.get_event_loop()
        segments, language = await loop.run_in_executor(_executor, _do_transcribe, tmp_path)

        return TranscribeResponse(
            task_id=task_id or None,
            language=language,
            full_text=" ".join(s.text for s in segments),
            segments=segments,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)
