import base64
import logging
from typing import List

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel, Field
from server.stream_status import read_stream_status

LOGGER = logging.getLogger("streamcat.http")


class InferHttpRequest(BaseModel):
    data_b64: str = Field(..., description="Base64 encoded contiguous tensor bytes")
    shape: List[int]
    dtype: str = "float32"


class InferHttpReply(BaseModel):
    data_b64: str
    shape: List[int]
    dtype: str
    infer_ms: float
    error: str = ""


def create_app(runtime, metrics, health_state, stream_status_file: str = "/tmp/streamcat_stream_status.json") -> FastAPI:
    app = FastAPI(title="StreamCat MONAI Service", version="0.1.0")
    templates = Jinja2Templates(directory="server/templates")

    @app.get("/")
    def read_root(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/health/live")
    def live():
        snap = health_state.snapshot()
        return {"live": snap.live, "message": snap.message}

    @app.get("/health/ready")
    def ready():
        snap = health_state.snapshot()
        return {"ready": snap.ready, "message": snap.message}

    @app.post("/infer", response_model=InferHttpReply)
    def infer(req: InferHttpRequest):
        try:
            raw = base64.b64decode(req.data_b64)
            x = np.frombuffer(raw, dtype=np.dtype(req.dtype)).reshape(tuple(req.shape))
            y, infer_ms = runtime.infer(x)
            metrics.infer_batch_size.observe(float(x.shape[0] if x.ndim >= 1 else 1))
            metrics.infer_latency_ms.observe(infer_ms)
            metrics.requests_total.labels(protocol="http", status="ok").inc()
            return InferHttpReply(
                data_b64=base64.b64encode(y.tobytes(order="C")).decode("ascii"),
                shape=list(y.shape),
                dtype=str(y.dtype),
                infer_ms=infer_ms,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("HTTP infer failed")
            metrics.requests_total.labels(protocol="http", status="error").inc()
            return InferHttpReply(data_b64="", shape=[], dtype="", infer_ms=0.0, error=str(exc))

    @app.get("/stream/status")
    def stream_status():
        return read_stream_status(stream_status_file)

    return app
