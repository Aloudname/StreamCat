from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class ServiceConfig:
    http_host: str
    http_port: int
    grpc_host: str
    grpc_port: int
    metrics_port: int
    log_level: str
    model_runtime: str
    model_path: str
    input_name: str
    output_name: str
    device: str
    batch_size: int


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_service_config(path: str) -> ServiceConfig:
    raw = _read_yaml(path)
    inf = raw.get("inference", {})
    srv = raw.get("server", {})

    return ServiceConfig(
        http_host=srv.get("http_host", "0.0.0.0"),
        http_port=int(srv.get("http_port", 8000)),
        grpc_host=srv.get("grpc_host", "0.0.0.0"),
        grpc_port=int(srv.get("grpc_port", 8001)),
        metrics_port=int(srv.get("metrics_port", 8002)),
        log_level=srv.get("log_level", "INFO"),
        model_runtime=inf.get("model_runtime", "onnx"),
        model_path=inf.get("model_path", ""),
        input_name=inf.get("input_name", "input"),
        output_name=inf.get("output_name", "output"),
        device=inf.get("device", "cuda"),
        batch_size=int(inf.get("batch_size", 64)),
    )
