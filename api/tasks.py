import json
import os
import subprocess
import time
from typing import Any, Dict, Optional

import noisereduce
import requests
import soundfile
from celery import Celery, chain
from celery.exceptions import TaskError

from .config import REDIS_URL, ASR_SERVICE_URL, UPLOAD_DIR

# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------
celery_app = Celery("worker", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# ---------------------------------------------------------------------------
# Redis helpers (raw hset, no ORM needed)
# ---------------------------------------------------------------------------
import redis as _redis

_redis_client = _redis.Redis.from_url(REDIS_URL, decode_responses=True)


def _set_status(task_id: str, status: str, result: Optional[Dict] = None) -> None:
    data: Dict[str, Any] = {"status": status, "updated_at": str(time.time())}
    if result:
        data["result"] = json.dumps(result)
    key = f"task:{task_id}"
    _redis_client.hset(key, mapping=data)
    _redis_client.expire(key, 3600)


def get_task_status(task_id: str) -> Dict[str, Any]:
    raw = _redis_client.hgetall(f"task:{task_id}")
    if not raw:
        return {"status": "not_found"}
    result = {}
    if "result" in raw:
        try:
            result = json.loads(raw["result"])
        except Exception:
            result = {"raw": raw["result"]}
    return {
        "status": raw.get("status", "unknown"),
        "updated_at": float(raw.get("updated_at", 0)),
        "result": result,
    }


def _send_callback(url: str, task_id: str, status: str, result: Optional[Dict] = None) -> None:
    if not url:
        return
    try:
        requests.post(
            url,
            json={"task_id": task_id, "status": status, "result": result or {}},
            timeout=5,
        )
    except Exception as exc:
        print(f"[callback] Task {task_id}: {exc}")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="convert", max_retries=3, default_retry_delay=10)
def ffmpeg_convert(self, task_id: str, input_path: str, callback_url: str = ""):
    """Конвертирует аудио в WAV 16 кГц моно через ffmpeg."""
    _set_status(task_id, "converting")
    output_path = f"{os.path.splitext(input_path)[0]}_converted.wav"
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise TaskError(f"ffmpeg: {proc.stderr}")
        return {"task_id": task_id, "input_path": input_path,
                "output_path": output_path, "callback_url": callback_url}
    except subprocess.TimeoutExpired as exc:
        _set_status(task_id, "failed", {"error": "conversion timeout"})
        raise self.retry(exc=exc)
    except Exception as exc:
        _set_status(task_id, "failed", {"error": str(exc)})
        raise self.retry(exc=exc)


@celery_app.task(bind=True, name="denoise", max_retries=3, default_retry_delay=10)
def denoise(self, prev: Dict[str, Any]):
    """Применяет шумоподавление (CPU)."""
    task_id = prev["task_id"]
    input_path = prev["output_path"]
    _set_status(task_id, "denoising")
    output_path = f"{os.path.splitext(input_path)[0]}_denoised.wav"
    try:
        audio, sr = soundfile.read(input_path)
        clean = noisereduce.reduce_noise(y=audio, sr=sr)
        soundfile.write(output_path, clean, sr)
        os.remove(input_path)
        return {**prev, "output_path": output_path}
    except Exception as exc:
        _set_status(task_id, "failed", {"error": str(exc)})
        raise self.retry(exc=exc)


@celery_app.task(bind=True, name="transcribe", max_retries=3, default_retry_delay=10)
def transcribe(self, prev: Dict[str, Any]):
    """Отправляет WAV в ASR-сервис и сохраняет транскрипцию."""
    task_id = prev["task_id"]
    input_path = prev["output_path"]
    callback_url = prev.get("callback_url", "")
    _set_status(task_id, "transcribing")
    try:
        with open(input_path, "rb") as f:
            resp = requests.post(
                f"{ASR_SERVICE_URL}/transcribe",
                files={"file": (os.path.basename(input_path), f, "audio/wav")},
                data={"task_id": task_id},
                timeout=30 * 60,
            )
        if resp.status_code != 200:
            raise TaskError(f"ASR error {resp.status_code}: {resp.text}")

        asr = resp.json()
        result = {
            "task_id": task_id,
            "transcription": asr.get("full_text", ""),
            "segments": asr.get("segments", []),
            "language": asr.get("language"),
        }
        _set_status(task_id, "completed", result)
        _send_callback(callback_url, task_id, "completed", result)
        return result
    except Exception as exc:
        error = {"error": str(exc)}
        _set_status(task_id, "failed", error)
        _send_callback(callback_url, task_id, "failed", error)
        raise self.retry(exc=exc)


@celery_app.task(name="process_audio")
def process_audio(task_id: str, file_path: str, callback_url: str = "") -> Dict[str, Any]:
    """Точка входа: запускает конвейер convert → denoise → transcribe."""
    _set_status(task_id, "queued")
    workflow = chain(
        ffmpeg_convert.s(task_id, file_path, callback_url),
        denoise.s(),
        transcribe.s(),
    )
    result = workflow.apply_async()
    return {"task_id": task_id, "chain_id": result.id}
