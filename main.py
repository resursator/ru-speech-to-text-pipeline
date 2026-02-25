import uuid
import json
import os
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException

from redis_client import redis_client
from config import REDIS_QUEUE, UPLOAD_DIR
from schemas import AudioTask

app = FastAPI(title="Audio Ingest API")

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    callback_url: str = "None"
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    task_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")

    # save file
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    task = AudioTask(
        task_id=task_id,
        filename=file.filename,
        path=save_path,
        created_at=datetime.utcnow().isoformat(),
        callback_url=callback_url
    )

    # push to redis queue
    redis_client.rpush(REDIS_QUEUE, json.dumps(task.model_dump()))

    return {
        "task_id": task_id,
        "status": "queued"
    }
