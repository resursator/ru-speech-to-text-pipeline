from pydantic import BaseModel
from datetime import datetime

class AudioTask(BaseModel):
    task_id: str
    filename: str
    path: str
    created_at: str
    status: str = "uploaded"
    callback_url: str = "None"
