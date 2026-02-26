FROM python:3.14-alpine

WORKDIR /app
RUN apk add --no-cache ffmpeg
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
