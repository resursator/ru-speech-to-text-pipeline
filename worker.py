import json
import time
from redis_client import redis_client
from config import REDIS_QUEUE

def process(task: dict):
    """
    Здесь будет:
    - ASR модель
    - транскрипция
    - вызов внешнего API
    """
    print(f"Processing task: {task['task_id']}")
    print(f"File: {task['path']}")

    # mock
    time.sleep(2)
    print("Done")


if __name__ == "__main__":
    print("Worker started")
    while True:
        _, raw = redis_client.blpop(REDIS_QUEUE)
        task = json.loads(raw)
        process(task)
