from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    task_id: str
    status: str
    created_at: str


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResult(BaseModel):
    task_id: str
    transcription: str
    segments: List[Segment]
    language: Optional[str]


class TaskStatusResponse(BaseModel):
    status: str
    updated_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
