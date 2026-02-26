import os
import subprocess
import json
import requests
from celery import Celery, chain
from celery.exceptions import TaskError
import time
from typing import Dict, Any, Optional

from config import REDIS_HOST, REDIS_PORT, UPLOAD_DIR
from redis_client import redis_client

# Создаем Celery приложение
celery_app = Celery(
    'audio_worker',
    broker=f'redis://{REDIS_HOST}:{REDIS_PORT}/0',
    backend=f'redis://{REDIS_HOST}:{REDIS_PORT}/0'
)

# Настройки Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 минут максимум
    task_soft_time_limit=25 * 60,  # мягкий лимит 25 минут
    task_acks_late=True,  # подтверждаем выполнение только после завершения
    task_reject_on_worker_lost=True,
    task_default_retry_delay=5,  # задержка перед повторной попыткой
    task_max_retries=3,  # максимум 3 попытки
    task_always_eager=False,  # не выполнять задачи синхронно
)

def update_task_status(task_id: str, status: str, result: Optional[Dict[str, Any]] = None):
    """Обновляет статус задачи в Redis"""
    try:
        task_key = f"task:{task_id}"
        task_data = {
            "status": status,
            "updated_at": str(time.time())
        }
        if result:
            task_data["result"] = json.dumps(result)

        redis_client.hset(task_key, mapping=task_data)
        redis_client.expire(task_key, 3600)  # TTL 1 час
    except Exception as e:
        print(f"Error updating status for task {task_id}: {e}")

def send_callback(callback_url: str, task_id: str, status: str, result: Optional[Dict[str, Any]] = None):
    """Отправляет callback с результатом обработки"""
    if callback_url and callback_url != "None":
        try:
            payload = {
                "task_id": task_id,
                "status": status,
                "result": result or {}
            }
            requests.post(callback_url, json=payload, timeout=5)
        except Exception as e:
            print(f"Callback failed for task {task_id}: {e}")

@celery_app.task(bind=True, name='convert')
def ffmpeg_convert(self, task_id, input_path, output_format='wav', callback_url="None"):
    """
    Конвертирует аудиофайл в указанный формат
    """
    update_task_status(task_id, "converting")

    try:
        # Создаем путь для выходного файла
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_converted.{output_format}"

        # Команда ffmpeg для конвертации
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-y',  # перезаписывать выходной файл
            '-acodec', 'pcm_s16le' if output_format == 'wav' else 'libmp3lame',
            '-ar', '16000',  # частота дискретизации 16kHz для ASR
            '-ac', '1',  # моно
            output_path
        ]

        # Выполняем конвертацию
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise TaskError(f"FFmpeg error: {result.stderr}")

        print(f"Task {task_id}: converted {input_path} to {output_path}")

        # Возвращаем данные для следующей задачи
        return {
            'task_id': task_id,
            'input_path': input_path,
            'output_path': output_path,
            'format': output_format,
            'callback_url': callback_url
        }

    except subprocess.TimeoutExpired:
        update_task_status(task_id, "failed", {"error": "Conversion timeout"})
        raise self.retry(exc=TaskError("Conversion timeout"), countdown=10)
    except Exception as e:
        update_task_status(task_id, "failed", {"error": str(e)})
        raise self.retry(exc=e, countdown=10)

@celery_app.task(bind=True, name='denoise')
def denoise(self, task_data):
    """
    Применяет шумоподавление к аудиофайлу
    """
    task_id = task_data['task_id']
    input_path = task_data['output_path']
    callback_url = task_data.get('callback_url', "None")

    update_task_status(task_id, "denoising")

    try:
        # Создаем путь для выходного файла
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_denoised.wav"

        # Здесь должна быть реальная реализация шумоподавления
        # Например, через noisereduce или внешнюю библиотеку
        # Пока имитируем работу
        time.sleep(2)

        # Для демо просто копируем файл
        import shutil
        shutil.copy2(input_path, output_path)

        print(f"Task {task_id}: denoised {input_path} -> {output_path}")

        task_data['denoised_path'] = output_path
        task_data['callback_url'] = callback_url
        return task_data

    except Exception as e:
        update_task_status(task_id, "failed", {"error": str(e)})
        raise self.retry(exc=e, countdown=10)

@celery_app.task(bind=True, name='transcribe')
def transcribe(self, task_data):
    """
    Выполняет транскрибацию аудио
    """
    task_id = task_data['task_id']
    input_path = task_data.get('denoised_path', task_data.get('output_path'))
    callback_url = task_data.get('callback_url', "None")

    update_task_status(task_id, "transcribing")

    try:
        # Здесь должна быть реальная ASR модель (например, Whisper)
        # Пока имитируем работу
        time.sleep(3)

        # Мок-транскрипция
        transcription = "Это пример транскрипции аудиофайла."

        result = {
            'task_id': task_id,
            'transcription': transcription,
            'processed_files': {
                'original': task_data.get('input_path'),
                'converted': task_data.get('output_path'),
                'denoised': task_data.get('denoised_path')
            }
        }

        update_task_status(task_id, "completed", result)

        # Отправляем callback
        if callback_url and callback_url != "None":
            send_callback(callback_url, task_id, "completed", result)

        print(f"Task {task_id}: transcription completed")
        return result

    except Exception as e:
        error_result = {"error": str(e)}
        update_task_status(task_id, "failed", error_result)

        # Отправляем callback об ошибке
        if callback_url and callback_url != "None":
            send_callback(callback_url, task_id, "failed", error_result)

        raise self.retry(exc=e, countdown=10)

@celery_app.task(name='process_audio')
def process_audio(task_id, file_path, callback_url="None"):
    """
    Основная задача, которая запускает цепочку обработки
    """
    update_task_status(task_id, "processing")

    try:
        # Создаем цепочку задач - здесь важно не использовать .set() с kwargs
        workflow = chain(
            ffmpeg_convert.s(task_id, file_path, 'wav', callback_url),
            denoise.s(),
            transcribe.s()
        )

        # Запускаем цепочку
        result = workflow.apply_async()

        return {
            "task_id": task_id,
            "status": "processing_started",
            "chain_id": result.id if result else None
        }

    except Exception as e:
        update_task_status(task_id, "failed", {"error": str(e)})
        if callback_url and callback_url != "None":
            send_callback(callback_url, task_id, "failed", {"error": str(e)})
        raise

# Функция для получения статуса задачи
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Получает статус задачи из Redis"""
    try:
        task_key = f"task:{task_id}"
        task_data = redis_client.hgetall(task_key)

        if not task_data:
            return {"status": "not_found"}

        # Декодируем байтовые ключи в строки
        decoded_data = {}
        for key, value in task_data.items():
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            decoded_data[key] = value

        result = {}
        if 'result' in decoded_data:
            try:
                result = json.loads(decoded_data['result'])
            except:
                result = {"raw": decoded_data['result']}

        return {
            "status": decoded_data.get('status', 'unknown'),
            "updated_at": float(decoded_data.get('updated_at', 0)),
            "result": result
        }

    except Exception as e:
        print(f"Error getting status for task {task_id}: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == '__main__':
    # Запуск воркера через командную строку
    print("Starting Celery worker...")
    celery_app.worker_main(argv=['worker', '--loglevel=info', '--concurrency=2'])
