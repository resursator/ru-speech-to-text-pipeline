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
import torch
import torchaudio
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from config import REDIS_HOST, REDIS_PORT, UPLOAD_DIR
from redis_client import redis_client

_model = None
_processor = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_torch_dtype = torch.float16 if _device == "cuda" else torch.float32

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

        audio, soundread = soundfile.read(input_path)
        clean = noisereduce.reduce_noise(y=audio, sr=soundread)
        soundfile.write(output_path, clean, soundread)

        print(f"Task {task_id}: denoised {input_path} -> {output_path}")

        # Удаляем временный файл
        os.remove(input_path)

        task_data['denoised_path'] = output_path
        task_data['callback_url'] = callback_url
        return task_data

    except Exception as e:
        update_task_status(task_id, "failed", {"error": str(e)})
        raise self.retry(exc=e, countdown=10)

@celery_app.task(bind=True, name='transcribe')
def transcribe(self, task_data):
    """
    Выполняет транскрибацию аудио с помощью Qwen3-ASR.
    Входной файл: WAV, 16 кГц, моно, очищен от шума.
    """
    global _model, _processor, _device, _torch_dtype

    task_id = task_data['task_id']
    input_path = task_data.get('denoised_path', task_data.get('output_path'))
    callback_url = task_data.get('callback_url', "None")

    update_task_status(task_id, "transcribing")

    try:
        # Загружаем модель при первом вызове
        if _model is None or _processor is None:
            print(f"Loading Qwen3-ASR model on {_device}...")
            model_id = "Qwen/Qwen3-ASR-1.7B"
            print(f"model_id {str(model_id)}, torch_dtype {str(_torch_dtype)}, device {str(_device)}, processor {str(_processor)}")
            _model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=_torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(_device)
            _processor = AutoProcessor.from_pretrained(model_id)
            print("Model loaded successfully")

        # Загружаем аудио
        waveform, sample_rate = torchaudio.load(input_path)

        # Приводим к моно, если вдруг не моно (хотя по условию должно быть)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ресемплируем до 16 кГц (на всякий случай)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Параметры сегментирования (фиксированные)
        chunk_length_s = 30  # длительность сегмента в секундах
        chunk_samples = chunk_length_s * sample_rate
        total_samples = waveform.shape[1]

        transcript_parts = []

        # Обрабатываем аудио сегментами
        for start_sample in range(0, total_samples, chunk_samples):
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = waveform[:, start_sample:end_sample]

            # Пропускаем тишину (опционально)
            if torch.max(torch.abs(chunk)) < 0.01:
                continue

            # Подготавливаем вход для модели
            inputs = _processor(
                raw_speech=chunk.squeeze().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            ).to(_device)

            # Генерируем транскрипцию
            with torch.no_grad():
                generated_ids = _model.generate(
                    **inputs,
                    max_new_tokens=256,
                    language="ru",        # можно параметризовать, пока русский
                    task="transcribe",
                    num_beams=5
                )

            text = _processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

            if text:
                transcript_parts.append(text)

        # Объединяем все части в одну строку
        full_transcript = " ".join(transcript_parts) if transcript_parts else ""

        result = {
            'task_id': task_id,
            'transcription': full_transcript,
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

        if callback_url and callback_url != "None":
            send_callback(callback_url, task_id, "failed", error_result)

        # Повторяем задачу при ошибке (до 3 попыток)
        raise self.retry(exc=e, countdown=10)

@celery_app.task(name='process_audio')
def process_audio(task_id, file_path, callback_url="None"):
    """
    Основная задача, которая запускает цепочку обработки
    """
    update_task_status(task_id, "processing")

    try:
        workflow = chain(
            ffmpeg_convert.s(task_id, file_path, 'wav', callback_url),
            denoise.s(),
            transcribe.s()
        )

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
