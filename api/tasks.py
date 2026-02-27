import json
import os
import subprocess
import time
from typing import Any, Dict, Optional

import noisereduce
import requests
import soundfile
from celery import Celery, chain
from celery.exceptions import SoftTimeLimitExceeded, TaskError
from celery.signals import task_failure, task_revoked

from .config import REDIS_URL, ASR_SERVICE_URL, UPLOAD_DIR

# ---------------------------------------------------------------------------
# ASR health check
# ---------------------------------------------------------------------------
ASR_HEALTH_TIMEOUT  = int(os.getenv("ASR_HEALTH_TIMEOUT",  str(15 * 60)))  # 15 минут
ASR_HEALTH_INTERVAL = int(os.getenv("ASR_HEALTH_INTERVAL", "10"))           # пауза между попытками, сек


def _wait_for_asr(timeout: int = ASR_HEALTH_TIMEOUT, interval: int = ASR_HEALTH_INTERVAL) -> None:
    """
    Блокирует выполнение до тех пор, пока ASR-сервис не вернёт {"status": "ok"}
    на GET /health. Если сервис не поднялся за `timeout` секунд — бросает RuntimeError.
    """
    url = f"{ASR_SERVICE_URL}/health"
    deadline = time.monotonic() + timeout
    attempt = 0

    while time.monotonic() < deadline:
        attempt += 1
        try:
            resp = requests.get(url, timeout=5)
            body = resp.json() if "application/json" in resp.headers.get("content-type", "") else {}
            asr_status = body.get("status", "")
            if resp.status_code == 200 and asr_status == "ok":
                print(f"[ASR health] OK after {attempt} attempt(s)")
                return
            if resp.status_code == 200 and asr_status == "loading":
                print(f"[ASR health] attempt {attempt}: модель загружается, ждём…")
            elif resp.status_code == 200 and asr_status == "error":
                raise RuntimeError(f"ASR model failed to load: {body.get('detail', 'unknown')}")
            else:
                print(f"[ASR health] attempt {attempt}: HTTP {resp.status_code}, body={resp.text[:80]}")
        except RuntimeError:
            raise
        except requests.RequestException as exc:
            print(f"[ASR health] attempt {attempt}: недоступен — {exc}")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))

    raise RuntimeError(
        f"ASR service at {url} did not become healthy within {timeout} seconds"
    )

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


def _get_callback_url(task_id: str) -> str:
    """Читает callback_url из Redis, сохранённый при постановке задачи."""
    return _redis_client.hget(f"task:{task_id}", "callback_url") or ""


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
# Celery signals — перехват failed / revoked на уровне воркера
# ---------------------------------------------------------------------------

@task_failure.connect
def on_task_failure(sender=None, task_id=None, exception=None, args=None,
                    kwargs=None, traceback=None, einfo=None, **kw):
    """
    Срабатывает, когда задача окончательно провалилась (все retry исчерпаны
    или исключение не подлежит retry). Гарантирует финальный callback.
    """
    # Получаем task_id приложения из первого аргумента задачи.
    # Соглашение: первый positional arg всегда task_id (строка UUID).
    app_task_id = (args or [None])[0]
    if not isinstance(app_task_id, str) or len(app_task_id) != 36:
        # Для цепочек первый аргумент может быть словарём prev-результата
        if isinstance(app_task_id, dict):
            app_task_id = app_task_id.get("task_id")
    if not app_task_id:
        return

    error = str(exception) if exception else "unknown error"
    _set_status(app_task_id, "failed", {"error": error})
    callback_url = _get_callback_url(app_task_id)
    _send_callback(callback_url, app_task_id, "failed", {"error": error})


@task_revoked.connect
def on_task_revoked(sender=None, request=None, terminated=False,
                    signum=None, expired=False, **kw):
    """
    Срабатывает при отзыве задачи (celery.control.revoke) или kill-сигнале.
    """
    if request is None:
        return
    args = request.args or []
    app_task_id = args[0] if args else None
    if isinstance(app_task_id, dict):
        app_task_id = app_task_id.get("task_id")
    if not app_task_id:
        return

    reason = "expired" if expired else ("terminated" if terminated else "revoked")
    error = {"error": f"task {reason}"}
    _set_status(app_task_id, "failed", error)
    callback_url = _get_callback_url(app_task_id)
    _send_callback(callback_url, app_task_id, "failed", error)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="convert", max_retries=3, default_retry_delay=10)
def ffmpeg_convert(self, task_id: str, input_path: str, callback_url: str = ""):
    """Конвертирует аудио в WAV 16 кГц моно через ffmpeg."""
    _set_status(task_id, "converting")
    _send_callback(callback_url, task_id, "converting")           # ← уведомление о старте

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

    except SoftTimeLimitExceeded:
        error = {"error": "conversion soft time limit exceeded"}
        _set_status(task_id, "failed", error)
        _send_callback(callback_url, task_id, "failed", error)
        raise                                                       # не retry — завершаем задачу

    except subprocess.TimeoutExpired as exc:
        # retry ещё возможен — не шлём финальный failed, signal on_task_failure доделает
        _set_status(task_id, "converting")                         # оставляем промежуточный статус
        raise self.retry(exc=exc)

    except TaskError as exc:
        # ffmpeg вернул ненулевой код — скорее всего повторная попытка не поможет,
        # но соблюдём max_retries
        raise self.retry(exc=exc)

    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(bind=True, name="denoise", max_retries=3, default_retry_delay=10)
def denoise(self, prev: Dict[str, Any]):
    """Применяет шумоподавление (CPU)."""
    task_id      = prev["task_id"]
    input_path   = prev["output_path"]
    callback_url = prev.get("callback_url", "")

    _set_status(task_id, "denoising")
    _send_callback(callback_url, task_id, "denoising")             # ← уведомление о старте

    output_path = f"{os.path.splitext(input_path)[0]}_denoised.wav"
    try:
        audio, sr = soundfile.read(input_path)
        clean = noisereduce.reduce_noise(y=audio, sr=sr)
        soundfile.write(output_path, clean, sr)
        os.remove(input_path)
        return {**prev, "output_path": output_path}

    except SoftTimeLimitExceeded:
        error = {"error": "denoising soft time limit exceeded"}
        _set_status(task_id, "failed", error)
        _send_callback(callback_url, task_id, "failed", error)
        raise

    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(bind=True, name="transcribe", max_retries=3, default_retry_delay=10)
def transcribe(self, prev: Dict[str, Any]):
    """Отправляет WAV в ASR-сервис и сохраняет транскрипцию."""
    task_id      = prev["task_id"]
    input_path   = prev["output_path"]
    callback_url = prev.get("callback_url", "")

    _set_status(task_id, "waiting_for_asr")
    _send_callback(callback_url, task_id, "waiting_for_asr")       # ← уведомление о старте ожидания ASR

    try:
        _wait_for_asr()
    except RuntimeError as exc:
        error = {"error": str(exc)}
        _set_status(task_id, "failed", error)
        _send_callback(callback_url, task_id, "failed", error)
        raise TaskError(str(exc))

    _set_status(task_id, "transcribing")
    _send_callback(callback_url, task_id, "transcribing")           # ← уведомление о старте транскрипции

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
        _send_callback(callback_url, task_id, "completed", result)  # ← финальный успех
        return result

    except SoftTimeLimitExceeded:
        error = {"error": "transcription soft time limit exceeded"}
        _set_status(task_id, "failed", error)
        _send_callback(callback_url, task_id, "failed", error)
        raise

    except TaskError as exc:
        # Не retry — ASR вернул явный код ошибки; сигнал on_task_failure добьёт
        raise

    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(name="process_audio")
def process_audio(task_id: str, file_path: str, callback_url: str = "") -> Dict[str, Any]:
    """Точка входа: сохраняет callback_url в Redis и запускает конвейер convert → denoise → transcribe."""
    # Сохраняем callback_url в Redis, чтобы сигналы task_failure / task_revoked
    # могли его прочитать, не имея доступа к аргументам цепочки
    _redis_client.hset(f"task:{task_id}", "callback_url", callback_url)
    _redis_client.expire(f"task:{task_id}", 3600)

    _set_status(task_id, "queued")
    _send_callback(callback_url, task_id, "queued")                 # ← уведомление о постановке в очередь

    workflow = chain(
        ffmpeg_convert.s(task_id, file_path, callback_url),
        denoise.s(),
        transcribe.s(),
    )
    result = workflow.apply_async()
    return {"task_id": task_id, "chain_id": result.id}
