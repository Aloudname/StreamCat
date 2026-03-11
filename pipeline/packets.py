# packets.py — Data packet definitions for the streaming pipeline.
#
# All inter-stage communication uses typed dataclass packets for
# type safety and automatic latency breakdown tracking.
#
# Packet flow:
#   CaptureSource -> FramePacket -> StreamPreprocessor -> InferPacket
#   -> InferClient -> InferPacket (with raw_output) -> TileStitcher
#   -> ResultPacket -> StreamDisplay

import numpy as np
from typing import Dict, Tuple, Any
from dataclasses import dataclass, field



@dataclass
class FramePacket:
    """Raw frame captured from the device.

    Attributes:
        frame_id:    Monotonically increasing frame counter.
        timestamp:   ``time.monotonic()`` at the moment of capture.
        image:       (H, W, C) numpy array, uint8 or float32 depending on source.
        camera_meta: Optional dict of device metadata (exposure, gain, etc.).
    """
    frame_id: int
    timestamp: float
    image: np.ndarray
    camera_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferPacket:
    """Preprocessed tiles ready for model inference.

    Attributes:
        frame_id:       Originating frame ID.
        timestamp:      Original capture timestamp (for end-to-end latency).
        tiles:          (N, C, H, W) float32 batch of patches.
        tile_coords:    (N, 2) int32 array; each row is (row, col) top-left
                        corner in the *padded* frame coordinate system.
        frame_shape:    (H, W) of the original (unpadded) frame.
        padded_shape:   (H, W) of the frame after reflection-padding.
        original_image: Reference to the raw BGR frame for overlay display.
        preprocess_ms:  Time spent in preprocessing (milliseconds).
        raw_output:     Filled by inference: (N, num_classes, H, W) float32.
        infer_ms:       Filled by inference: time in inference (milliseconds).
    """
    frame_id: int
    timestamp: float
    tiles: np.ndarray
    tile_coords: np.ndarray
    frame_shape: Tuple[int, int]
    padded_shape: Tuple[int, int]
    original_image: np.ndarray
    preprocess_ms: float = 0.0
    raw_output: np.ndarray = None
    infer_ms: float = 0.0


@dataclass
class ResultPacket:
    """Final result ready for display.

    Attributes:
        frame_id:           Originating frame ID.
        timestamp:          Original capture timestamp.
        original_image:     (H, W, C) BGR frame for overlay.
        segmentation_mask:  (H, W) int32, per-pixel predicted class index.
        confidence_map:     (H, W) float32, max softmax probability per pixel.
        class_probs:        (num_classes, H, W) float32 full probability map.
        latency:            Per-stage latency breakdown (ms).
    """
    frame_id: int
    timestamp: float
    original_image: np.ndarray
    segmentation_mask: np.ndarray
    confidence_map: np.ndarray
    class_probs: np.ndarray = None
    latency: Dict[str, float] = field(default_factory=dict)
