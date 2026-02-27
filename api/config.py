import os

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_URL  = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

# Storage
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")

# ASR
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://asr:8001")
