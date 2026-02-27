FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_VERSION=2.10.0
RUN pip install --no-cache-dir [ -z "$TORCH_VERSION" ] && \
    echo "TORCH_VERSION не задан" ||\
    pip install --index-url https://download.pytorch.org/whl torch==${TORCH_VERSION}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
