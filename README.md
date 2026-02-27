# Audio Transcription Service

Сервис транскрипции звонков. Принимает аудиофайл, конвертирует, применяет шумоподавление и возвращает текст + временные метки внешней системе через callback.

## Архитектура

```
Client → API (FastAPI) → Redis Queue → Worker (Celery) → ASR Service
                                                      ↓
                                              Callback URL (внешняя система)
```

| Компонент | Роль |
|-----------|------|
| `api` | Приём файлов, выдача статуса задач |
| `worker` | Конвертация (ffmpeg), шумоподавление (noisereduce), вызов ASR |
| `asr` | Транскрипция через Whisper (CPU) или Qwen3-ASR (GPU) |
| `redis` | Брокер задач + хранение статусов |

## Режимы запуска

### CPU (faster-whisper, без GPU)

```bash
# Собрать и запустить
docker compose -f docker-compose.yml up --build

# Отправить файл на транскрипцию
curl -X POST http://localhost:8000/upload \
  -F "file=@call.mp3" \
  -F "callback_url=https://your-system.example.com/webhook"

# Проверить статус
curl http://localhost:8000/tasks/<task_id>
```

Используемая модель: `faster-whisper` (`medium` по умолчанию, можно менять через `ASR_MODEL_SIZE`).  
Первый запуск загружает модель из HuggingFace (~1.5 ГБ для `medium`).

### GPU (Qwen3-ASR, требует NVIDIA + nvidia-container-toolkit)

```bash
# Убедитесь, что nvidia-container-toolkit установлен:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Собрать и запустить
docker compose -f docker-compose-gpu.yml up --build

# Отправить файл на транскрипцию
curl -X POST http://localhost:8000/upload \
  -F "file=@call.mp3" \
  -F "callback_url=https://your-system.example.com/webhook"
```

Используемая модель: `Qwen/Qwen3-ASR-1.7B` (по умолчанию). Первый запуск загружает модель (~3.5 ГБ).

## API

### `POST /upload`
Принимает аудиофайл (mp3, wav, ogg, …) и опциональный `callback_url`.

**Response 202:**
```json
{ "task_id": "uuid", "status": "queued", "created_at": "2024-01-01T00:00:00" }
```

### `GET /tasks/{task_id}`
Возвращает статус задачи.

**Статусы:** `queued` → `converting` → `denoising` → `transcribing` → `completed` / `failed`

**Response (completed):**
```json
{
  "status": "completed",
  "result": {
    "transcription": "Текст звонка…",
    "segments": [{"start": 0.0, "end": 30.0, "text": "…"}],
    "language": "Russian"
  }
}
```

### Callback (внешняя система)

При завершении сервис делает `POST callback_url`:
```json
{
  "task_id": "uuid",
  "status": "completed",
  "result": { "transcription": "…", "segments": […] }
}
```
При ошибке: `"status": "failed"`, `"result": {"error": "…"}`.

## Масштабирование

```bash
# CPU
docker compose -f docker-compose.yml up --scale worker=4

# GPU
docker compose -f docker-compose-gpu.yml up --scale worker=4
```

ASR-сервис работает как отдельный под — можно вынести на выделенный GPU-узел без изменения остального кода.

## Переменные окружения

### API / Worker

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `REDIS_HOST` | `redis` | Хост Redis |
| `REDIS_PORT` | `6379` | Порт Redis |
| `UPLOAD_DIR` | `/app/uploads` | Папка для файлов |
| `ASR_SERVICE_URL` | `http://asr:8001` | URL ASR-сервиса |
| `ASR_HEALTH_TIMEOUT` | `900` | Макс. ожидание старта ASR, сек |
| `ASR_HEALTH_INTERVAL` | `10` | Пауза между проверками ASR, сек |

### ASR — CPU (`docker-compose.yml`)

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `ASR_MODEL_SIZE` | `medium` | Размер модели: `tiny` / `base` / `small` / `medium` / `large-v3` |
| `ASR_LANGUAGE` | `ru` | Язык (`auto` = определять автоматически) |
| `ASR_COMPUTE_TYPE` | `int8` | Тип вычислений (int8 быстрее на CPU) |
| `ASR_BEAM_SIZE` | `5` | Beam size при декодировании |
| `ASR_CPU_THREADS` | `4` | Потоки CPU |

### ASR — GPU (`docker-compose-gpu.yml`)

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `ASR_MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace модель |
| `ASR_LANGUAGE` | `Russian` | Язык (`auto` = определять автоматически) |
| `ASR_CHUNK_S` | `30` | Длина сегмента в секундах |
| `ASR_MAX_TOKENS` | `256` | Макс. токенов на сегмент |
