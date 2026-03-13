import time
from typing import Callable, Tuple

import numpy as np


class MonaiRuntime:
    def __init__(
        self,
        model_runtime: str,
        model_path: str,
        input_name: str,
        output_name: str,
        device: str,
    ) -> None:
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
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers and self._device.type == "cuda":
                selected = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                selected = ["CPUExecutionProvider"]

            session = ort.InferenceSession(self._model_path, providers=selected)

            def _onnx_network(x):
                x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
                y_np = session.run([self._output_name], {self._input_name: x_np})[0]
                import torch
                return torch.from_numpy(y_np).to(x.device)

            self._network = _onnx_network
        elif self._runtime == "torchscript":
            import torch

            model = torch.jit.load(self._model_path, map_location=self._device)
            model.eval()

            def _torchscript_network(x):
                return model(x)

            self._network = _torchscript_network
        else:
            raise ValueError(f"Unsupported runtime: {self._runtime}")

    def infer(self, x_np: np.ndarray) -> Tuple[np.ndarray, float]:
        import torch

        t0 = time.monotonic()
        x = torch.from_numpy(x_np.astype(np.float32, copy=False)).to(self._device)
        with torch.no_grad():
            y = self._inferer(x, self._network)
        infer_ms = (time.monotonic() - t0) * 1000.0
        return y.detach().cpu().numpy(), infer_ms
