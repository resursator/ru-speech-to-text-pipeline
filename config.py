import os

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "audio_tasks")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
