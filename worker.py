import os
import subprocess
import json
import requests
from celery import Celery, chain
from celery.exceptions import TaskError
import time
from typing import Dict, Any, Optional
import noisereduce
import soundfile

from config import REDIS_HOST, REDIS_PORT, UPLOAD_DIR
from redis_client import redis_client

# ---------------------------------------------------------------------------
# ASR-сервис
# ---------------------------------------------------------------------------
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://asr:8001")

# ---------------------------------------------------------------------------
# Celery
# ---------------------------------------------------------------------------
celery_app = Celery(
    'audio_worker',
    broker=f'redis://{REDIS_HOST}:{REDIS_PORT}/0',
    backend=f'redis://{REDIS_HOST}:{REDIS_PORT}/0',
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=5,
    task_max_retries=3,
    task_always_eager=False,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def update_task_status(task_id: str, status: str, result: Optional[Dict[str, Any]] = None):
    try:
        task_key  = f"task:{task_id}"
        task_data = {"status": status, "updated_at": str(time.time())}
        if result:
            task_data["result"] = json.dumps(result)
        redis_client.hset(task_key, mapping=task_data)
        redis_client.expire(task_key, 3600)
    except Exception as e:
        print(f"Error updating status for task {task_id}: {e}")


def send_callback(callback_url: str, task_id: str, status: str, result: Optional[Dict[str, Any]] = None):
    if callback_url and callback_url != "None":
        try:
            requests.post(
                callback_url,
                json={"task_id": task_id, "status": status, "result": result or {}},
                timeout=5,
            )
        except Exception as e:
            print(f"Callback failed for task {task_id}: {e}")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name='convert')
def ffmpeg_convert(self, task_id: str, input_path: str, output_format: str = 'wav', callback_url: str = "None"):
    """Конвертирует аудио в WAV 16кГц моно через ffmpeg."""
    update_task_status(task_id, "converting")
    try:
        base_name   = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_converted.{output_format}"

        cmd = [
            'ffmpeg', '-i', input_path, '-y',
            '-acodec', 'pcm_s16le' if output_format == 'wav' else 'libmp3lame',
            '-ar', '16000',
            '-ac', '1',
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise TaskError(f"FFmpeg error: {result.stderr}")

        print(f"Task {task_id}: converted {input_path} → {output_path}")
        return {
            'task_id':      task_id,
            'input_path':   input_path,
            'output_path':  output_path,
            'format':       output_format,
            'callback_url': callback_url,
        }

    except subprocess.TimeoutExpired:
        update_task_status(task_id, "failed", {"error": "Conversion timeout"})
        raise self.retry(exc=TaskError("Conversion timeout"), countdown=10)
    except Exception as e:
        update_task_status(task_id, "failed", {"error": str(e)})
        raise self.retry(exc=e, countdown=10)


@celery_app.task(bind=True, name='denoise')
def denoise(self, task_data: Dict[str, Any]):
    """Применяет шумоподавление (CPU)."""
    task_id      = task_data['task_id']
    input_path   = task_data['output_path']
    callback_url = task_data.get('callback_url', "None")

    update_task_status(task_id, "denoising")
    try:
        base_name   = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_denoised.wav"

        audio, sr = soundfile.read(input_path)
        clean     = noisereduce.reduce_noise(y=audio, sr=sr)
        soundfile.write(output_path, clean, sr)

        os.remove(input_path)   # удаляем временный конвертированный файл
        print(f"Task {task_id}: denoised → {output_path}")

        task_data['denoised_path'] = output_path
        task_data['callback_url']  = callback_url
        return task_data

    except Exception as e:
        update_task_status(task_id, "failed", {"error": str(e)})
        raise self.retry(exc=e, countdown=10)


@celery_app.task(bind=True, name='transcribe')
def transcribe(self, task_data: Dict[str, Any]):
    """
    Отправляет денойзнутый WAV в ASR-сервис (отдельный GPU-контейнер)
    и сохраняет результат.
    """
    task_id      = task_data['task_id']
    input_path   = task_data.get('denoised_path', task_data.get('output_path'))
    callback_url = task_data.get('callback_url', "None")

    update_task_status(task_id, "transcribing")
    try:
        url = f"{ASR_SERVICE_URL}/transcribe"
        with open(input_path, 'rb') as f:
            resp = requests.post(
                url,
                files={"file": (os.path.basename(input_path), f, "audio/wav")},
                data={"task_id": task_id},
                # таймаут большой — длинные файлы обрабатываются долго
                timeout=60 * 30,
            )

        if resp.status_code != 200:
            raise TaskError(f"ASR service error {resp.status_code}: {resp.text}")

        asr_result = resp.json()

        result = {
            'task_id':       task_id,
            'transcription': asr_result.get('full_text', ''),
            'segments':      asr_result.get('segments', []),
            'language':      asr_result.get('language'),
            'processed_files': {
                'original': task_data.get('input_path'),
                'converted': task_data.get('output_path'),
                'denoised':  input_path,
            },
        }

        update_task_status(task_id, "completed", result)
        send_callback(callback_url, task_id, "completed", result)
        print(f"Task {task_id}: transcription completed")
        return result

    except Exception as e:
        error_result = {"error": str(e)}
        update_task_status(task_id, "failed", error_result)
        send_callback(callback_url, task_id, "failed", error_result)
        raise self.retry(exc=e, countdown=10)


@celery_app.task(name='process_audio')
def process_audio(task_id: str, file_path: str, callback_url: str = "None"):
    """Запускает цепочку: convert → denoise → transcribe."""
    update_task_status(task_id, "processing")
    try:
        workflow = chain(
            ffmpeg_convert.s(task_id, file_path, 'wav', callback_url),
            denoise.s(),
            transcribe.s(),
        )
        result = workflow.apply_async()
        return {
            "task_id":  task_id,
            "status":   "processing_started",
            "chain_id": result.id if result else None,
        }
    except Exception as e:
        update_task_status(task_id, "failed", {"error": str(e)})
        send_callback(callback_url, task_id, "failed", {"error": str(e)})
        raise


# ---------------------------------------------------------------------------
# Status helper
# ---------------------------------------------------------------------------

def get_task_status(task_id: str) -> Dict[str, Any]:
    try:
        task_data = redis_client.hgetall(f"task:{task_id}")
        if not task_data:
            return {"status": "not_found"}

        decoded = {
            (k.decode() if isinstance(k, bytes) else k):
            (v.decode() if isinstance(v, bytes) else v)
            for k, v in task_data.items()
        }

        result = {}
        if 'result' in decoded:
            try:
                result = json.loads(decoded['result'])
            except Exception:
                result = {"raw": decoded['result']}

        return {
            "status":     decoded.get('status', 'unknown'),
            "updated_at": float(decoded.get('updated_at', 0)),
            "result":     result,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == '__main__':
    print("Starting Celery worker (no GPU)...")
    celery_app.worker_main(argv=['worker', '--loglevel=info', '--concurrency=2'])
