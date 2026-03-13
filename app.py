#!/usr/bin/env python3
import argparse
import logging

import uvicorn

from server.config import load_service_config
from server.grpc_service import start_grpc_server
from server.health import HealthState
from server.http_api import create_app
from server.logging_setup import setup_logging
from server.metrics import ServiceMetrics
from server.model_runtime import MonaiRuntime


LOGGER = logging.getLogger("streamcat.app")


def parse_args():
    parser = argparse.ArgumentParser(description="StreamCat MONAI service")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="Path to config YAML")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_service_config(args.config)

    setup_logging(cfg.log_level)
    health = HealthState()
    metrics = ServiceMetrics()

    LOGGER.info("starting metrics on :%s", cfg.metrics_port)
    try:
        metrics.start_metrics_server(cfg.metrics_port)
    except OSError as exc:
        LOGGER.error(
            "failed to bind metrics port %s: %s. "
            "A previous process may still be running (often after Ctrl+Z). "
            "Stop it and retry.",
            cfg.metrics_port,
            exc,
        )
        raise

    runtime = MonaiRuntime(
        model_runtime=cfg.model_runtime,
        model_path=cfg.model_path,
        input_name=cfg.input_name,
        output_name=cfg.output_name,
        device=cfg.device,
    )

    health.set(ready=True, message="ready")
    metrics.ready_state.set(1)

    grpc_server = start_grpc_server(
        runtime=runtime,
        metrics=metrics,
        health_state=health,
        host=cfg.grpc_host,
        port=cfg.grpc_port,
        max_message_mb=cfg.grpc_max_message_mb,
    )

    app = create_app(
        runtime=runtime,
        metrics=metrics,
        health_state=health,
        stream_status_file=cfg.stream_status_file,
    )
    try:
        LOGGER.info("starting HTTP on %s:%s", cfg.http_host, cfg.http_port)
        uvicorn.run(app, host=cfg.http_host, port=cfg.http_port, log_level=cfg.log_level.lower())
    finally:
        LOGGER.info("shutting down services")
        health.set(live=False, ready=False, message="stopping")
        metrics.ready_state.set(0)
        grpc_server.stop(grace=2)


if __name__ == "__main__":
    main()
