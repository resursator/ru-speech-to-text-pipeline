FROM python:3.11-slim

ARG UID=1000
ARG GID=1000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/

RUN groupadd -g ${GID} appgroup \
    && useradd -u ${UID} -g appgroup -m appuser \
    && chown -R ${UID}:${GID} /app
USER appuser

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EXPOSE 8000
