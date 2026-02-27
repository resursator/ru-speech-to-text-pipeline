import uuid
import os
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from .config import UPLOAD_DIR
from .schemas import UploadResponse, TaskStatusResponse
from .tasks import process_audio, get_task_status

app = FastAPI(title="Audio Transcription API")

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_audio(
    file: UploadFile = File(...),
    callback_url: str = "",
):
    """
    Принимает аудиофайл, сохраняет и ставит задачу на транскрипцию в очередь.
    Возвращает task_id для последующей проверки статуса.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    task_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".bin"
    save_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")

    with open(save_path, "wb") as f:
        f.write(await file.read())

    process_audio.delay(task_id, save_path, callback_url)

    return UploadResponse(task_id=task_id, status="queued", created_at=datetime.utcnow().isoformat())


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def task_status(task_id: str):
    """Возвращает текущий статус и результат задачи транскрипции."""
    data = get_task_status(task_id)
    if data["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Task not found")
    return data


@app.get("/health")
async def health():
    return {"status": "ok"}
