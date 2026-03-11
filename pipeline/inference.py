# inference.py — Triton / ONNX Runtime inference execution layer.
#
# Sends InferPacket tiles to the model backend in batched chunks
# and fills the ``raw_output`` / ``infer_ms`` fields in-place.
#
# Two backends:
#   - "triton": Triton Inference Server via gRPC (preferred in production).
#   - "onnx":   Local ONNX Runtime session (for dev / debugging without Docker).
#
# Usage::
#
#     client = InferClient.create(cfg.inference)
#     client.infer(infer_packet)          # fills raw_output in-place
#     assert client.health_check()

import time
import numpy as np

from abc import ABC, abstractmethod
from typing import Optional
from munch import Munch
from pipeline.monitor import tprint


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
            A TritonClient or OnnxClient instance.
        """
        backend = cfg.backend.lower()
        if backend == "triton":
            return TritonClient(
                url=cfg.triton_url,
                protocol=cfg.triton_protocol,
                model_name=cfg.model_name,
                model_version=cfg.get("model_version", ""),
                input_name=cfg.input_name,
                output_name=cfg.output_name,
                timeout_ms=cfg.timeout_ms,
            )
        elif backend == "onnx":
            return OnnxClient(
                onnx_path=cfg.onnx_path,
                input_name=cfg.input_name,
                output_name=cfg.output_name,
            )
        else:
            raise ValueError(f"Unknown inference backend: '{backend}'. "
                             "Use 'triton' or 'onnx'.")


class TritonClient(InferClient):
    """Triton Inference Server client (gRPC or HTTP).

    gRPC is preferred for lower per-request overhead.
    """

    def __init__(self,
                 url: str = "localhost:8001",
                 protocol: str = "grpc",
                 model_name: str = "common_mini",
                 model_version: str = "",
                 input_name: str = "input",
                 output_name: str = "output",
                 timeout_ms: int = 5000):
        self._url = url
        self._protocol = protocol.lower()
        self._model_name = model_name
        self._model_version = model_version
        self._input_name = input_name
        self._output_name = output_name
        self._timeout = timeout_ms / 1000.0  # convert to seconds

        self._client = None
        self._client_module = None
        self._connect()

    def _connect(self) -> None:
        """Lazily import tritonclient and create the connection."""
        if self._protocol == "grpc":
            try:
                import tritonclient.grpc as grpcclient
            except ImportError:
                raise ImportError(
                    "tritonclient[grpc] is required for Triton gRPC backend. "
                    "Install with: pip install tritonclient[grpc]")
            self._client_module = grpcclient
            self._client = grpcclient.InferenceServerClient(url=self._url)
        else:
            try:
                import tritonclient.http as httpclient
            except ImportError:
                raise ImportError(
                    "tritonclient[http] is required for Triton HTTP backend. "
                    "Install with: pip install tritonclient[http]")
            self._client_module = httpclient
            self._client = httpclient.InferenceServerClient(url=self._url)
        tprint(f"[inference] triton {self._protocol} connected: {self._url}")

    def infer(self, tiles: np.ndarray) -> np.ndarray:
        """Send a batch to Triton and return the prediction array.

        Args:
            tiles: (N, C, H, W) float32.

        Returns:
            (N, num_classes, H, W) float32.
        """
        mod = self._client_module
        inp = mod.InferInput(self._input_name, list(tiles.shape), "FP32")
        inp.set_data_from_numpy(tiles)

        out = mod.InferRequestedOutput(self._output_name)
        result = self._client.infer(
            model_name=self._model_name,
            model_version=self._model_version,
            inputs=[inp],
            outputs=[out],
        )
        return result.as_numpy(self._output_name)

    def health_check(self) -> bool:
        try:
            return self._client.is_server_live()
        except Exception:
            return False


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
