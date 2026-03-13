# inference.py — MONAI-centered inference execution layer.
#
# Sends InferPacket tiles to MONAI/ONNX backends in batched chunks
# and fills the ``raw_output`` / ``infer_ms`` fields in-place.
#
# Two backends:
#   - "monai": MONAI inferer + runtime adapter (onnx / torchscript).
#   - "onnx":  Local ONNX Runtime direct session (fallback).
#   - "grpc":  Remote MONAI service via gRPC.
#
# Usage::
#
#     client = InferClient.create(cfg.inference)
#     client.infer(infer_packet)          # fills raw_output in-place
#     assert client.health_check()

import numpy as np

from abc import ABC, abstractmethod
from typing import Callable
from munch import Munch
from pipeline.monitor import tprint
from server.proto import infer_pb2_grpc


class InferClient(ABC):
    """Abstract base for inference backends."""

    @abstractmethod
    def infer(self, tiles: np.ndarray) -> np.ndarray:
        """Run inference on a (N, C, H, W) float32 batch.

        Args:
            tiles: Input tensor.

        Returns:
            (N, num_classes, H, W) float32 prediction logits / probabilities.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the backend is reachable and ready."""

    @staticmethod
    def create(cfg: Munch) -> 'InferClient':
        """Factory: build a concrete client from streaming config.

        Args:
            cfg: ``config.inference`` Munch section.

        Returns:
            A MonaiClient or OnnxClient instance.
        """
        backend = cfg.backend.lower()
        if backend == "monai":
            return MonaiClient(
                model_runtime=cfg.get("model_runtime", "onnx"),
                model_path=cfg.model_path,
                input_name=cfg.get("input_name", "input"),
                output_name=cfg.get("output_name", "output"),
                device=cfg.get("device", "cuda"),
            )
        elif backend == "grpc":
            return GrpcClient(
                target=cfg.get("grpc_target", "localhost:8001"),
                max_message_mb=cfg.get("grpc_max_message_mb", 64),
            )
        elif backend == "onnx":
            return OnnxClient(
                onnx_path=cfg.model_path,
                input_name=cfg.get("input_name", "input"),
                output_name=cfg.get("output_name", "output"),
            )
        else:
            raise ValueError(f"Unknown inference backend: '{backend}'. "
                             "Use 'monai', 'onnx', or 'grpc'.")


class GrpcClient(InferClient):
    """Remote gRPC client for StreamCat MONAI service."""

    def __init__(self, target: str = "localhost:8001", max_message_mb: int = 64):
        try:
            import grpc
        except ImportError:
            raise ImportError("grpcio is required for gRPC backend")

        from server.proto_gen import ensure_proto_generated

        ensure_proto_generated()
        from server.proto import infer_pb2

        self._grpc = grpc
        self._pb2 = infer_pb2
        max_bytes = int(max_message_mb) * 1024 * 1024
        self._channel = grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", max_bytes),
                ("grpc.max_receive_message_length", max_bytes),
            ],
        )
        self._stub = infer_pb2_grpc.InferenceServiceStub(self._channel)
        self._target = target
        tprint(f"[inference] gRPC backend connected: {target} (max_message_mb={max_message_mb})")

    def infer(self, tiles: np.ndarray) -> np.ndarray:
        request = self._pb2.InferRequest(
            request_id="stream-pipeline",
            input=self._pb2.Tensor(
                data=tiles.astype(np.float32, copy=False).tobytes(order="C"),
                shape=list(tiles.shape),
                dtype=str(tiles.dtype),
            ),
        )
        reply = self._stub.Infer(request)
        if reply.error:
            raise RuntimeError(f"gRPC infer failed: {reply.error}")
        out = np.frombuffer(reply.output.data, dtype=np.dtype(reply.output.dtype))
        return out.reshape(tuple(reply.output.shape))

    def health_check(self) -> bool:
        try:
            reply = self._stub.Health(self._pb2.HealthRequest())
            return bool(reply.live and reply.ready)
        except Exception:
            return False


class MonaiClient(InferClient):
    """MONAI inference client with runtime adapters.

    Supports:
        - onnx runtime via ONNX Runtime
        - torchscript runtime via torch.jit
    """

    def __init__(self,
                 model_runtime: str = "onnx",
                 model_path: str = "",
                 input_name: str = "input",
                 output_name: str = "output",
                 device: str = "cuda"):
        self._runtime = model_runtime.lower()
        self._model_path = model_path
        self._input_name = input_name
        self._output_name = output_name
        self._network: Callable = None

        import torch
        from monai.inferers import SimpleInferer

        if device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._inferer = SimpleInferer()
        self._build_network()

    def _build_network(self) -> None:
        if self._runtime == "onnx":
            try:
                import onnxruntime as ort
            except ImportError:
                raise ImportError(
                    "MONAI onnx runtime requires onnxruntime or onnxruntime-gpu. "
                    "Install with: pip install onnxruntime-gpu")

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers and self._device.type == "cuda":
                selected = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                selected = ["CPUExecutionProvider"]

            session = ort.InferenceSession(self._model_path, providers=selected)
            tprint(f"[inference] MONAI backend loaded ONNX model: {self._model_path}")
            tprint(f"[inference] ONNX providers={selected}")

            def _onnx_network(x):
                x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
                y_np = session.run([self._output_name], {self._input_name: x_np})[0]
                import torch
                return torch.from_numpy(y_np).to(x.device)

            self._network = _onnx_network

        elif self._runtime == "torchscript":
            try:
                import torch
            except ImportError:
                raise ImportError(
                    "MONAI torchscript runtime requires PyTorch. "
                    "Install with: pip install torch")

            model = torch.jit.load(self._model_path, map_location=self._device)
            model.eval()
            tprint(f"[inference] MONAI backend loaded TorchScript model: {self._model_path}")

            def _torchscript_network(x):
                return model(x)

            self._network = _torchscript_network

        else:
            raise ValueError(
                f"Unsupported MONAI runtime '{self._runtime}'. "
                "Use 'onnx' or 'torchscript'."
            )

    def infer(self, tiles: np.ndarray) -> np.ndarray:
        """Run MONAI inference and return the prediction array.

        Args:
            tiles: (N, C, H, W) float32.

        Returns:
            (N, num_classes, H, W) float32.
        """
        import torch

        x = torch.from_numpy(tiles.astype(np.float32, copy=False)).to(self._device)
        with torch.no_grad():
            y = self._inferer(x, self._network)
        return y.detach().cpu().numpy()

    def health_check(self) -> bool:
        return self._network is not None


class OnnxClient(InferClient):
    """Local ONNX Runtime inference for dev / debugging without a server.

    Loads a ``.onnx`` model and runs on CPU (or CUDA if available).
    """

    def __init__(self,
                 onnx_path: str,
                 input_name: str = "input",
                 output_name: str = "output"):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX backend. "
                "Install with: pip install onnxruntime-gpu  (or onnxruntime)")

        providers = ort.get_available_providers()
        # prefer CUDA, fall back to CPU
        if "CUDAExecutionProvider" in providers:
            selected = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            selected = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(onnx_path, providers=selected)
        self._input_name = input_name
        self._output_name = output_name
        tprint(f"[inference] onnx loaded: {onnx_path}  providers={selected}")

    def infer(self, tiles: np.ndarray) -> np.ndarray:
        """Run local ONNX inference.

        Args:
            tiles: (N, C, H, W) float32.

        Returns:
            (N, num_classes, H, W) float32.
        """
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: tiles},
        )
        return outputs[0]

    def health_check(self) -> bool:
        return self._session is not None


def batched_infer(client: InferClient,
                  tiles: np.ndarray,
                  batch_size: int = 64) -> np.ndarray:
    """Run inference in chunks of ``batch_size`` tiles.

    Prevents GPU OOM by splitting large tile arrays and concatenating results.

    Args:
        client:     InferClient instance.
        tiles:      (N, C, H, W) float32.
        batch_size: Max tiles per call.

    Returns:
        (N, num_classes, H, W) float32, concatenated.
    """
    N = tiles.shape[0]
    if N <= batch_size:
        return client.infer(tiles)

    parts = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        parts.append(client.infer(tiles[start:end]))
    return np.concatenate(parts, axis=0)
