"""
Демо-стенд для Audio Transcription Service.
- Мониторинг здоровья API и ASR сервисов
- Загрузка файла и отслеживание задачи (polling каждые 5 сек)
- Callback-приёмник (POST /callback)
- Кэширование статусов задач (TTL=4 сек) для снижения нагрузки на бэкенд
"""

import os
import time
import threading
from collections import OrderedDict
from datetime import datetime
from typing import Optional

import gradio as gr
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

# ── Конфигурация ──────────────────────────────────────────────────────────────
API_URL  = os.getenv("API_URL",  "http://api:8000")
ASR_URL  = os.getenv("ASR_URL",  "http://asr:8001")
DEMO_URL = os.getenv("DEMO_URL", "http://demo:7860")


# ── Простой TTL-кэш для статусов задач ───────────────────────────────────────
class TTLCache:
    def __init__(self, ttl: float = 4.0, maxsize: int = 256):
        self._cache: OrderedDict = OrderedDict()
        self._ttl     = ttl
        self._maxsize = maxsize
        self._lock    = threading.Lock()

    def get(self, key: str):
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            value, ts = entry
            if time.monotonic() - ts > self._ttl:
                del self._cache[key]
                return None
            return value

    def set(self, key: str, value):
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = (value, time.monotonic())


_task_cache = TTLCache(ttl=4.0)

# ── Хранилище callback-уведомлений (in-memory, последние 50) ─────────────────
_callbacks: list[dict] = []
_cb_lock = threading.Lock()

def _add_callback(data: dict):
    with _cb_lock:
        _callbacks.append({"received_at": datetime.utcnow().isoformat(), **data})
        if len(_callbacks) > 50:
            _callbacks.pop(0)

def _get_callbacks() -> list[dict]:
    with _cb_lock:
        return list(reversed(_callbacks))


# ── HTTP-хелперы ──────────────────────────────────────────────────────────────
def _check_health(url: str, name: str) -> tuple[bool, str]:
    try:
        r = requests.get(f"{url}/health", timeout=3)
        if r.status_code == 200:
            data  = r.json()
            extra = ""
            if "model"  in data: extra  = f" · модель: {data['model']}"
            if "device" in data: extra += f" · устройство: {data['device']}"
            return True, f"✅ {name}: работает{extra}"
        return False, f"⚠️ {name}: HTTP {r.status_code}"
    except Exception as e:
        return False, f"❌ {name}: недоступен ({e})"


def _get_task_status(task_id: str) -> dict:
    cached = _task_cache.get(task_id)
    if cached is not None:
        return cached
    try:
        r    = requests.get(f"{API_URL}/tasks/{task_id}", timeout=5)
        data = r.json() if r.status_code == 200 else {"status": "error", "result": {"error": r.text}}
    except Exception as e:
        data = {"status": "error", "result": {"error": str(e)}}
    _task_cache.set(task_id, data)
    return data


# ── Форматирование ────────────────────────────────────────────────────────────
STATUS_LABELS = {
    "queued":          "в очереди",
    "converting":      "конвертация",
    "denoising":       "шумоподавление",
    "waiting_for_asr": "ожидание ASR",
    "transcribing":    "транскрипция",
    "completed":       "завершено",
    "failed":          "ошибка",
    "error":           "ошибка",
}

STATUS_ICONS = {
    "queued":          "🟡",
    "converting":      "🔵",
    "denoising":       "🔵",
    "waiting_for_asr": "🔵",
    "transcribing":    "🔵",
    "completed":       "🟢",
    "failed":          "🔴",
    "error":           "🔴",
}

def fmt_status(s: str) -> str:
    icon  = STATUS_ICONS.get(s, "⚪")
    label = STATUS_LABELS.get(s, s)
    return f"{icon} {label}"

def fmt_result(data: dict) -> str:
    result = data.get("result") or {}
    status = data.get("status", "")
    if status == "completed":
        text = result.get("transcription", "")
        lang = result.get("language", "")
        segs = result.get("segments", [])
        lines = [f"**Язык:** {lang}", f"**Транскрипция:**\n\n{text}", ""]
        if segs:
            lines.append("**Сегменты:**")
            for seg in segs:
                lines.append(f"- `[{seg['start']:.1f}s – {seg['end']:.1f}s]` {seg['text']}")
        return "\n".join(lines)
    if status in ("failed", "error"):
        return f"**Ошибка:** {result.get('error', 'неизвестно')}"
    return ""

def fmt_callbacks(items: list[dict]) -> str:
    if not items:
        return "*Уведомлений ещё не поступало.*"
    lines = []
    for cb in items[:10]:
        ts      = cb.get("received_at", "")[:19].replace("T", " ")
        tid     = cb.get("task_id", "?")[:8]
        st      = cb.get("status", "?")
        icon    = STATUS_ICONS.get(st, "⚪")
        label   = STATUS_LABELS.get(st, st)
        result  = cb.get("result") or {}
        preview = (result.get("transcription") or result.get("error") or "")[:80]
        if preview:
            preview = f" — _{preview}…_" if len(preview) == 80 else f" — _{preview}_"
        lines.append(f"`{ts}` · **{tid}…** · {icon} {label}{preview}")
    return "\n\n".join(lines)


# ── Действия UI ───────────────────────────────────────────────────────────────
def check_services():
    _, api_msg = _check_health(API_URL, "API")
    _, asr_msg = _check_health(ASR_URL, "ASR")
    return api_msg, asr_msg


MIME_MAP = {
    ".mp3": "audio/mpeg",        ".wav": "audio/wav",        ".ogg": "audio/ogg",
    ".flac": "audio/flac",       ".aac": "audio/aac",        ".m4a": "audio/mp4",
    ".wma": "audio/x-ms-wma",    ".opus": "audio/opus",
    ".mp4": "video/mp4",         ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",   ".webm": "video/webm",
    ".ts":  "video/mp2t",        ".mts": "video/mp2t",       ".m2ts": "video/mp2t",
}

def upload_file(audio_file, use_callback: bool):
    if audio_file is None:
        return "⚠️ Выберите файл", "", gr.update(visible=False)
    file_path    = audio_file if isinstance(audio_file, str) else audio_file.name
    callback_url = f"{DEMO_URL}/callback" if use_callback else ""
    ext  = os.path.splitext(file_path)[1].lower()
    mime = MIME_MAP.get(ext, "application/octet-stream")
    try:
        with open(file_path, "rb") as f:
            fname = os.path.basename(file_path)
            r = requests.post(
                f"{API_URL}/upload",
                files={"file": (fname, f, mime)},
                params={"callback_url": callback_url} if callback_url else None,
                timeout=30,
            )
        if r.status_code in (200, 202):
            data    = r.json()
            task_id = data["task_id"]
            return task_id, fmt_status("queued"), gr.update(visible=True)
        return "", f"❌ Ошибка загрузки: HTTP {r.status_code} — {r.text}", gr.update(visible=False)
    except Exception as e:
        return "", f"❌ {e}", gr.update(visible=False)


def poll_status(task_id: str):
    if not task_id:
        return "", "", ""
    data       = _get_task_status(task_id)
    status     = data.get("status", "unknown")
    status_str = fmt_status(status)
    result_str = fmt_result(data)
    updated    = data.get("updated_at")
    ts = f"Обновлено: {datetime.utcfromtimestamp(updated).strftime('%H:%M:%S')} UTC" if updated else ""
    return status_str, result_str, ts


def refresh_callbacks():
    return fmt_callbacks(_get_callbacks())


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Unbounded:wght@300;700&display=swap');

:root {
    --bg:      #0d0f14;
    --surface: #161922;
    --border:  #252a35;
    --accent:  #00e5ff;
    --accent2: #7c3aed;
    --text:    #e2e8f0;
    --muted:   #64748b;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

h1, h2, h3 { font-family: 'Unbounded', sans-serif !important; letter-spacing: -0.02em; }

.gr-panel, .gr-box, .gr-block, .gr-form {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

#header {
    text-align: center;
    padding: 2rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
#header h1 {
    font-size: 1.6rem;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#header p { color: var(--muted); font-size: 0.8rem; margin-top: 0.3rem; }
"""

# ── Сборка интерфейса ─────────────────────────────────────────────────────────
# Gradio 6: css передаётся в gr.Blocks, theme — тоже, но используем встроенный Base
# без явного указания, чтобы избежать конфликтов версий.
with gr.Blocks(css=CSS, title="Демо — Транскрипция аудио") as demo:

    gr.HTML("""
    <div id="header">
        <h1>⚡ Сервис транскрипции звонков — Демо</h1>
        <p>Распознавание речи в реальном времени · Шумоподавление · Callback-уведомления</p>
    </div>
    """)

    # ── Статус сервисов ────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🖥 Состояние сервисов")
            with gr.Row():
                api_status = gr.Markdown("*Проверяется…*")
                asr_status = gr.Markdown("*Проверяется…*")
            btn_refresh_health = gr.Button("↻ Обновить статус", variant="secondary", size="sm")

    btn_refresh_health.click(check_services, outputs=[api_status, asr_status])
    demo.load(check_services, outputs=[api_status, asr_status])

    gr.HTML("<hr style='border-color:#252a35; margin: 1.5rem 0'>")

    # ── Загрузка и транскрипция ────────────────────────────────────────────
    gr.Markdown("### 🎙 Транскрипция")
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.File(
                label="Аудио или видеофайл",
                file_types=[
                    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".opus",
                    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".mts", ".m2ts",
                ],
            )
            use_cb = gr.Checkbox(
                label=f"Включить callback-уведомления → {DEMO_URL}/callback",
                value=True,
            )
            btn_upload = gr.Button("▶ Отправить на транскрипцию", variant="primary")

        with gr.Column(scale=2):
            task_id_box = gr.Textbox(
                label="Идентификатор задачи",
                interactive=False,
                placeholder="—",
            )
            task_status_box = gr.Markdown("*Ожидание загрузки…*")
            task_ts_box     = gr.Markdown("")
            result_box      = gr.Markdown("")
            poll_row        = gr.Row(visible=False)
            with poll_row:
                btn_poll = gr.Button("↻ Обновить статус задачи", variant="secondary", size="sm")

    btn_upload.click(
        upload_file,
        inputs=[audio_input, use_cb],
        outputs=[task_id_box, task_status_box, poll_row],
    )

    btn_poll.click(
        poll_status,
        inputs=[task_id_box],
        outputs=[task_status_box, result_box, task_ts_box],
    )

    # Авто-поллинг каждые 5 секунд
    timer_poll = gr.Timer(5)
    timer_poll.tick(
        poll_status,
        inputs=[task_id_box],
        outputs=[task_status_box, result_box, task_ts_box],
    )

    gr.HTML("<hr style='border-color:#252a35; margin: 1.5rem 0'>")

    # ── Callback-лента ─────────────────────────────────────────────────────
    gr.Markdown("### 📡 Входящие callback-уведомления")
    cb_feed = gr.Markdown("*Уведомлений ещё не поступало.*")
    btn_cb_refresh = gr.Button("↻ Обновить ленту", variant="secondary", size="sm")

    btn_cb_refresh.click(refresh_callbacks, outputs=[cb_feed])
    timer_cb = gr.Timer(5)
    timer_cb.tick(refresh_callbacks, outputs=[cb_feed])
    demo.load(refresh_callbacks, outputs=[cb_feed])


# ── FastAPI: монтируем Gradio + добавляем /callback endpoint ─────────────────
app = FastAPI()

@app.post("/callback")
async def callback(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    _add_callback(data)
    return JSONResponse({"ok": True})

@app.get("/")
async def root():
    return RedirectResponse(url="/ui")

# Градио монтируем на /ui, чтобы /callback не конфликтовал с Gradio-роутами
app = gr.mount_gradio_app(app, demo, path="/ui")
