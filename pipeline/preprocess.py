# preprocess.py — Frame preprocessing and tiling for streaming inference.
#
# Responsibilities:
#   1. Normalize pixel values (simple /255 for RGB, z-score+PCA for HS).
#   2. Pad frame with reflection to ensure full tile coverage.
#   3. Sliding-window extraction of (patch_size x patch_size) tiles.
#   4. Channel adaptation when camera channels != model channels.
#   5. Batch assembly into a single (N, C, H, W) float32 tensor.
#
# Usage::
#
#     prep = StreamPreprocessor(patch_size=31, stride=16)
#     infer_packet = prep.process(frame_packet)

import time
import pickle
import numpy as np

from typing import Optional
from pipeline.monitor import tprint
from pipeline.packets import FramePacket, InferPacket



class StreamPreprocessor:
    """Converts raw FramePackets into batched InferPackets.

    Handles normalization, channel adaptation, reflection-padded tiling,
    and HWC -> NCHW transposition in a single pass.

    The ``normalize_mode`` controls how raw pixel values are scaled:

    - ``"simple"``: divide by 255.0 (suitable for RGB testing).
    - ``"hs"``: load a pickled ``HSPreprocessor`` (z-score + PCA)
      used during training, ensuring train-inference consistency.
    """

    def __init__(self,
                 patch_size: int = 31,
                 stride: int = 16,
                 normalize_mode: str = "simple",
                 preprocessor_path: str = "",
                 model_channels: Optional[int] = None):
        """
        Args:
            patch_size: Tile spatial size (must match model training).
            stride:     Sliding-window step. Smaller = more overlap = smoother.
            normalize_mode: ``"simple"`` or ``"hs"``.
            preprocessor_path: Path to a pickled HSPreprocessor (only for ``"hs"``).
            model_channels: Expected input channels for the model.
                            If None, inferred from the preprocessor or camera.
        """
        self._patch_size = patch_size
        self._stride = stride
        self._normalize_mode = normalize_mode
        self._model_channels = model_channels
        self._hs_preprocessor = None

        if normalize_mode == "hs" and preprocessor_path:
            with open(preprocessor_path, "rb") as f:
                self._hs_preprocessor = pickle.load(f)
            tprint(f"[preprocess] loaded HSPreprocessor from {preprocessor_path}")

    def process(self, frame: FramePacket) -> InferPacket:
        """Convert a single FramePacket into an InferPacket of tiled patches.

        Steps:
            1. Normalize pixel values.
            2. Adapt channel dimension if needed.
            3. Reflection-pad to ensure full tile coverage.
            4. Extract overlapping tiles with sliding window.
            5. Transpose to NCHW and pack into InferPacket.

        Args:
            frame: Raw camera frame.

        Returns:
            InferPacket with batched tiles ready for inference.
        """
        t0 = time.monotonic()

        image = frame.image  # (H, W, C), uint8 or float32

        # 1. normalize
        image = self._normalize(image)

        # 2. channel adapt
        image = self._adapt_channels(image)

        # 3. reflection-pad for full coverage
        H, W, C = image.shape
        pad_h = self._compute_pad(H)
        pad_w = self._compute_pad(W)
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image,
                           ((0, pad_h), (0, pad_w), (0, 0)),
                           mode="reflect")
        padded_H, padded_W = image.shape[:2]

        # 4. sliding-window tiling
        tiles, coords = self._tile(image)

        # 5. HWC -> NCHW
        tiles = tiles.transpose(0, 3, 1, 2).astype(np.float32)

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        preview = frame.camera_meta.get("preview_image", frame.image)

        return InferPacket(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            tiles=tiles,
            tile_coords=coords,
            frame_shape=(H, W),
            padded_shape=(padded_H, padded_W),
            original_image=preview,
            preprocess_ms=elapsed_ms,
        )

    # ---- internals ----

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize pixel values to float32."""
        if self._normalize_mode == "hs" and self._hs_preprocessor is not None:
            # HSPreprocessor expects (n_pixels, n_bands) or (H, W, n_bands)
            return self._hs_preprocessor.transform(image.astype(np.float32))

        # simple: [0, 255] uint8 -> [0.0, 1.0] float32
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)

    def _adapt_channels(self, image: np.ndarray) -> np.ndarray:
        """Pad or truncate channels to match model expectation."""
        if self._model_channels is None:
            return image

        C = image.shape[2] if image.ndim == 3 else 1
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        if C == self._model_channels:
            return image
        elif C < self._model_channels:
            # zero-pad extra channels
            pad_c = self._model_channels - C
            pad = np.zeros((*image.shape[:2], pad_c), dtype=image.dtype)
            return np.concatenate([image, pad], axis=2)
        else:
            # truncate (unlikely in practice)
            return image[:, :, :self._model_channels]

    def _compute_pad(self, length: int) -> int:
        """Compute reflection-pad needed so last tile fits exactly."""
        ps = self._patch_size
        st = self._stride
        if length < ps:
            return ps - length
        remainder = (length - ps) % st
        return (st - remainder) % st

    def _tile(self, image: np.ndarray):
        """Extract overlapping tiles using a sliding window.

        Args:
            image: (H, W, C) padded float32 frame.

        Returns:
            tiles:  (N, patch_size, patch_size, C) float32 array.
            coords: (N, 2) int32 array of (row, col) top-left positions.
        """
        H, W, C = image.shape
        ps = self._patch_size
        st = self._stride

        rows = list(range(0, H - ps + 1, st))
        cols = list(range(0, W - ps + 1, st))

        n_tiles = len(rows) * len(cols)
        tiles = np.empty((n_tiles, ps, ps, C), dtype=image.dtype)
        coords = np.empty((n_tiles, 2), dtype=np.int32)

        idx = 0
        for r in rows:
            for c in cols:
                tiles[idx] = image[r:r + ps, c:c + ps, :]
                coords[idx] = (r, c)
                idx += 1

        return tiles, coords
