# postprocess.py — Tile stitching and segmentation mask generation.
#
# Reassembles per-tile model predictions into a full-frame segmentation
# mask with overlap fusion and optional morphological cleanup.
#
# Supports two fusion modes:
#   - "average":  Simple mean over overlapping tile predictions.
#   - "gaussian": Weighted by a 2-D Gaussian centered on each tile
#                 (reduces edge artefacts at tile boundaries).
#
# Usage::
#
#     stitcher = TileStitcher(patch_size=31, stride=16, num_classes=2)
#     result = stitcher.stitch(infer_packet)

import cv2
import time
import numpy as np

from typing import Optional
from pipeline.monitor import tprint
from pipeline.packets import InferPacket, ResultPacket



class TileStitcher:
    """Stitches prediction tiles back to full-frame resolution.

    After inference, each tile has shape (num_classes, patch_size, patch_size).
    This class reassembles them into a single (num_classes, H, W) probability
    map, then derives the argmax class mask and confidence map.
    """

    def __init__(self,
                 patch_size: int = 31,
                 stride: int = 16,
                 num_classes: int = 2,
                 fusion_mode: str = "average",
                 confidence_threshold: float = 0.3,
                 morphology_kernel: int = 0):
        """
        Args:
            patch_size:  Tile spatial size.
            stride:      Sliding-window step used during tiling.
            num_classes:  Number of output classes.
            fusion_mode: ``"average"`` or ``"gaussian"``.
            confidence_threshold: Below this, pixel is marked uncertain (255).
            morphology_kernel: Odd kernel size for morphological opening.
                               0 = disabled.
        """
        self._patch_size = patch_size
        self._stride = stride
        self._num_classes = num_classes
        self._fusion_mode = fusion_mode
        self._conf_thresh = confidence_threshold
        self._morph_k = morphology_kernel

        # Pre-compute Gaussian weight kernel (reused every frame)
        self._gauss_weight = self._build_gaussian_weight() if fusion_mode == "gaussian" else None

    def stitch(self, packet: InferPacket) -> ResultPacket:
        """Assemble tile predictions into a full-frame result.

        Args:
            packet: InferPacket with ``raw_output`` filled by inference.

        Returns:
            ResultPacket ready for display.
        """
        t0 = time.monotonic()

        preds = packet.raw_output       # (N, num_classes, ps, ps)
        coords = packet.tile_coords     # (N, 2) top-left (row, col) in padded space
        pH, pW = packet.padded_shape
        H, W = packet.frame_shape

        # Softmax over class dimension (axis=1) for probability maps
        preds = self._softmax(preds)

        # Accumulate weighted predictions
        accum = np.zeros((self._num_classes, pH, pW), dtype=np.float64)
        count = np.zeros((pH, pW), dtype=np.float64)

        weight = self._gauss_weight if self._gauss_weight is not None else 1.0

        ps = self._patch_size
        for i, (r, c) in enumerate(coords):
            accum[:, r:r + ps, c:c + ps] += preds[i] * weight
            count[r:r + ps, c:c + ps] += weight

        # Avoid division by zero
        count = np.maximum(count, 1e-8)
        prob_map = (accum / count[np.newaxis, :, :]).astype(np.float32)

        # Crop back to original (unpadded) size
        prob_map = prob_map[:, :H, :W]

        # Derive class mask and confidence
        seg_mask = prob_map.argmax(axis=0).astype(np.int32)      # (H, W)
        conf_map = prob_map.max(axis=0)                           # (H, W)

        # Mark low-confidence pixels as uncertain
        seg_mask[conf_map < self._conf_thresh] = 255

        # Optional morphological cleanup
        if self._morph_k > 0:
            seg_mask = self._morphology_clean(seg_mask)

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        latency = {
            "preprocess_ms": packet.preprocess_ms,
            "infer_ms": packet.infer_ms,
            "postprocess_ms": elapsed_ms,
            "total_ms": (time.monotonic() - packet.timestamp) * 1000.0,
        }

        return ResultPacket(
            frame_id=packet.frame_id,
            timestamp=packet.timestamp,
            original_image=packet.original_image,
            segmentation_mask=seg_mask,
            confidence_map=conf_map,
            class_probs=prob_map,
            latency=latency,
        )

    # ---- internals ----

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over axis=1 (class dimension).

        Args:
            x: (N, C, H, W)

        Returns:
            (N, C, H, W) with sum-to-one over C.
        """
        x_max = x.max(axis=1, keepdims=True)
        e = np.exp(x - x_max)
        return e / e.sum(axis=1, keepdims=True)

    def _build_gaussian_weight(self) -> np.ndarray:
        """2-D Gaussian kernel for smooth tile fusion.

        Pixels near tile center get higher weight, reducing
        boundary artefacts in the stitched result.
        """
        ps = self._patch_size
        ax = np.arange(ps, dtype=np.float32) - (ps - 1) / 2.0
        sigma = ps / 4.0
        kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.max()  # peak = 1.0
        return kernel_2d

    def _morphology_clean(self, mask: np.ndarray) -> np.ndarray:
        """Morphological opening to remove small noise regions."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._morph_k, self._morph_k))
        cleaned = mask.copy()
        for cls_id in range(self._num_classes):
            binary = (cleaned == cls_id).astype(np.uint8)
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            # restore pixels that survived opening
            cleaned[(binary == 1) & (opened == 0)] = 255
        return cleaned
