import json
import os
import tempfile
import threading
import time
from typing import Any, Dict


DEFAULT_STATUS_FILE = "/tmp/streamcat_stream_status.json"


class StreamStatusWriter:
    """Cross-process status writer for run.py pipeline progress."""

    def __init__(self, path: str = DEFAULT_STATUS_FILE):
        self._path = path or DEFAULT_STATUS_FILE
        self._lock = threading.Lock()
        self._last_t = None
        self._fps = 0.0
        self._started_at = time.time()

    @property
    def path(self) -> str:
        return self._path

    def mark_started(self, payload: Dict[str, Any]) -> None:
        self._started_at = time.time()
        self._last_t = None
        self._fps = 0.0
        base = {
            "running": True,
            "started_at": self._started_at,
            "updated_at": self._started_at,
            "frame_id": 0,
            "throughput_fps": 0.0,
            "recent_infer_ms": 0.0,
            "error": "",
        }
        base.update(payload or {})
        self._write(base)

    def update_frame(self, frame_id: int, infer_ms: float, extra: Dict[str, Any] = None) -> None:
        now = time.time()
        with self._lock:
            if self._last_t is not None:
                dt = max(now - self._last_t, 1e-6)
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0.0 else (0.8 * self._fps + 0.2 * inst)
            self._last_t = now

        data = read_stream_status(self._path)
        data.update({
            "running": True,
            "updated_at": now,
            "frame_id": int(frame_id),
            "throughput_fps": float(self._fps),
            "recent_infer_ms": float(infer_ms),
            "error": "",
        })
        if extra:
            data.update(extra)
        self._write(data)

    def mark_error(self, message: str) -> None:
        data = read_stream_status(self._path)
        data.update({
            "updated_at": time.time(),
            "error": str(message),
        })
        self._write(data)

    def mark_stopped(self, reason: str = "") -> None:
        data = read_stream_status(self._path)
        data.update({
            "running": False,
            "updated_at": time.time(),
            "stop_reason": reason,
        })
        self._write(data)

    def _write(self, payload: Dict[str, Any]) -> None:
        parent = os.path.dirname(self._path) or "."
        os.makedirs(parent, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=parent, delete=False) as tf:
            json.dump(payload, tf, ensure_ascii=True)
            tmp_name = tf.name
        os.replace(tmp_name, self._path)


def read_stream_status(path: str = DEFAULT_STATUS_FILE) -> Dict[str, Any]:
    p = path or DEFAULT_STATUS_FILE
    if not os.path.exists(p):
        return {
            "running": False,
            "frame_id": 0,
            "throughput_fps": 0.0,
            "recent_infer_ms": 0.0,
            "error": "no stream status yet",
            "status_file": p,
        }
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return {
            "running": False,
            "frame_id": 0,
            "throughput_fps": 0.0,
            "recent_infer_ms": 0.0,
            "error": f"failed to read status: {exc}",
            "status_file": p,
        }
    data.setdefault("status_file", p)
    return data
