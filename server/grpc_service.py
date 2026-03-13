import logging
from concurrent import futures

import grpc
import numpy as np

from server.proto_gen import ensure_proto_generated

ensure_proto_generated()
from server import infer_pb2, infer_pb2_grpc  # noqa: E402


LOGGER = logging.getLogger("streamcat.grpc")


def _tensor_to_ndarray(tensor: infer_pb2.Tensor) -> np.ndarray:
    dtype = np.dtype(tensor.dtype)
    arr = np.frombuffer(tensor.data, dtype=dtype)
    if tensor.shape:
        arr = arr.reshape(tuple(tensor.shape))
    return arr


def _ndarray_to_tensor(arr: np.ndarray) -> infer_pb2.Tensor:
    return infer_pb2.Tensor(
        data=arr.tobytes(order="C"),
        shape=list(arr.shape),
        dtype=str(arr.dtype),
    )


class InferenceServicer(infer_pb2_grpc.InferenceServiceServicer):
    def __init__(self, runtime, metrics, health_state):
        self._runtime = runtime
        self._metrics = metrics
        self._health = health_state

    def Infer(self, request, context):
        try:
            x = _tensor_to_ndarray(request.input)
            self._metrics.infer_batch_size.observe(float(x.shape[0] if x.ndim >= 1 else 1))
            y, infer_ms = self._runtime.infer(x)
            self._metrics.infer_latency_ms.observe(infer_ms)
            self._metrics.requests_total.labels(protocol="grpc", status="ok").inc()
            return infer_pb2.InferReply(
                output=_ndarray_to_tensor(y),
                infer_ms=infer_ms,
                error="",
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("gRPC infer failed")
            self._metrics.requests_total.labels(protocol="grpc", status="error").inc()
            return infer_pb2.InferReply(error=str(exc))

    def Health(self, request, context):
        snap = self._health.snapshot()
        return infer_pb2.HealthReply(
            live=snap.live,
            ready=snap.ready,
            message=snap.message,
        )


def start_grpc_server(runtime, metrics, health_state, host: str, port: int, workers: int = 8):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    infer_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(runtime, metrics, health_state),
        server,
    )
    bind_addr = f"{host}:{port}"
    server.add_insecure_port(bind_addr)
    server.start()
    LOGGER.info("gRPC server listening on %s", bind_addr)
    return server
