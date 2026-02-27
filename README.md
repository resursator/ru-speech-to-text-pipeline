# Audio Transcription Service

Сервис транскрипции звонков на базе Qwen3-ASR. Принимает аудиофайл, конвертирует, применяет шумоподавление и возвращает текст + временные метки внешней системе через callback.

## Архитектура

```
Client → API (FastAPI) → Redis Queue → Worker (Celery) → ASR Service (GPU)
                                                      ↓
                                              Callback URL (внешняя система)
```

| Компонент | Роль |
|-----------|------|
| `api` | Приём файлов, выдача статуса задач |
| `worker` | Конвертация (ffmpeg), шумоподавление (noisereduce), вызов ASR |
| `asr` | Транскрипция через Qwen3-ASR (GPU) |
| `redis` | Брокер задач + хранение статусов |

Компоненты слабо связаны: API не импортирует код worker'а напрямую, общение только через Redis.

## Запуск

```bash
# Собрать и запустить
docker compose up --build

# Отправить файл на транскрипцию
curl -X POST http://localhost:8000/upload \
  -F "file=@call.mp3" \
  -F "callback_url=https://your-system.example.com/webhook"

# Проверить статус
curl http://localhost:8000/tasks/<task_id>
```

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

Для увеличения пропускной способности достаточно запустить больше реплик worker'а — все они читают из одной Redis-очереди:

```bash
docker compose up --scale worker=4
```

ASR-сервис работает как отдельный под — можно вынести на выделенный GPU-узел без изменения остального кода.

## Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `REDIS_HOST` | `redis` | Хост Redis |
| `REDIS_PORT` | `6379` | Порт Redis |
| `UPLOAD_DIR` | `/app/uploads` | Папка для файлов |
| `ASR_SERVICE_URL` | `http://asr:8001` | URL ASR-сервиса |
| `ASR_MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace модель |
| `ASR_LANGUAGE` | `Russian` | Язык (`auto` = определять автоматически) |
| `ASR_CHUNK_S` | `30` | Длина сегмента в секундах |
