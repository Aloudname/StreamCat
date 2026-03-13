# capture.py — Device capture layer.
#
# Wraps camera / video file into a uniform FramePacket producer.
# A dedicated capture thread grabs frames at device FPS, storing
# only the latest frame so consumers never block on stale data.
#
# Supports:
#   - USB webcam (source = int device index)
#   - Video file (source = str file path)
#   - Reconnection on transient failures
#
# Usage::
#
#     src = CaptureSource(source=0, width=640, height=480, fps=30)
#     src.open()
#     frame = src.get_frame()   # returns FramePacket or None
#     src.release()

import cv2
import time
import threading
import numpy as np
import pathlib

from typing import Optional, Union
from pipeline.packets import FramePacket
from pipeline.monitor import tprint


class CaptureSource:
    """Camera or video capture with latest-frame semantics.

    A background thread continuously grabs frames from the device.
    ``get_frame()`` always returns the *most recent* frame, discarding
    any older buffered frames to maintain real-time freshness.
    """

    def __init__(self,
                 source: Union[int, str] = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 input_mode: str = "opencv",
                 npy_dir: str = "",
                 npy_glob: str = "*.npy",
                 npy_loop: bool = True,
                 npy_fps: float = 10.0,
                 reconnect_attempts: int = 5,
                 reconnect_delay_sec: float = 1.0):
        """
        Args:
            source: Device index (int) or video file path (str).
            width:  Capture width in pixels.
            height: Capture height in pixels.
            fps:    Target capture frame rate.
            reconnect_attempts: Max consecutive reconnect retries.
            reconnect_delay_sec: Delay between reconnect attempts.
        """
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._input_mode = input_mode
        self._npy_dir = npy_dir
        self._npy_glob = npy_glob
        self._npy_loop = npy_loop
        self._npy_fps = npy_fps
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay_sec

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Latest-frame buffer (protected by lock)
        self._lock = threading.Lock()
        self._latest_frame: Optional[FramePacket] = None
        self._frame_counter = 0
        self._npy_files = []
        self._npy_idx = 0

    def open(self) -> bool:
        """Open the capture device and start the grab thread.

        Returns:
            True if the device was opened successfully.
        """
        if self._input_mode == "npy_stream":
            npy_root = pathlib.Path(self._npy_dir) if self._npy_dir else None
            if npy_root is None or not npy_root.exists():
                tprint(f"[capture] npy_dir does not exist: {self._npy_dir}")
                return False
            self._npy_files = sorted(npy_root.glob(self._npy_glob))
            if not self._npy_files:
                tprint(f"[capture] no npy files found in {npy_root} with pattern '{self._npy_glob}'")
                return False

            tprint(f"[capture] opened npy stream: {len(self._npy_files)} files @ {self._npy_fps:.2f} fps")
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._npy_loop_worker, daemon=True,
                                            name="NpyCaptureThread")
            self._thread.start()
            return True

        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            tprint(f"[capture] FAILED to open source: {self._source}")
            return False

        # Configure device (ignored silently for video files)
        if isinstance(self._source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        tprint(f"[capture] opened: {actual_w}x{actual_h} @ {actual_fps:.1f} fps "
               f"(source={self._source})")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._grab_loop, daemon=True,
                                        name="CaptureThread")
        self._thread.start()
        return True

    def get_frame(self) -> Optional[FramePacket]:
        """Return the most recent frame, or None if nothing available."""
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None  # consumed
            return frame

    def release(self) -> None:
        """Stop the capture thread and release the device."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        tprint("[capture] released")

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # internal methods
    def _grab_loop(self) -> None:
        """Continuously grab frames on a background thread."""
        consecutive_failures = 0

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._try_reconnect():
                    break
                continue

            ret, frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > self._reconnect_attempts:
                    tprint("[capture] max failures reached, stopping")
                    break
                # For video file: loop back to beginning
                if isinstance(self._source, str):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            consecutive_failures = 0
            self._frame_counter += 1
            image, preview = self._adapt_camera_frame(frame)

            packet = FramePacket(
                frame_id=self._frame_counter,
                timestamp=time.monotonic(),
                image=image,
                camera_meta={
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "channels": image.shape[2] if image.ndim == 3 else 1,
                    "preview_image": preview,
                },
            )

            with self._lock:
                self._latest_frame = packet

    def _try_reconnect(self) -> bool:
        """Attempt to reopen the device after a failure."""
        for attempt in range(1, self._reconnect_attempts + 1):
            if self._stop_event.is_set():
                return False
            tprint(f"[capture] reconnect attempt {attempt}/{self._reconnect_attempts}")
            time.sleep(self._reconnect_delay)
            if self._cap:
                self._cap.release()
            self._cap = cv2.VideoCapture(self._source)
            if self._cap.isOpened():
                tprint("[capture] reconnected")
                return True
        tprint("[capture] reconnect failed, giving up")
        return False

    def _npy_loop_worker(self) -> None:
        """Continuously emit frames loaded from .npy files as a pseudo-stream."""
        period = 1.0 / max(self._npy_fps, 0.1)
        while not self._stop_event.is_set():
            if self._npy_idx >= len(self._npy_files):
                if self._npy_loop:
                    self._npy_idx = 0
                else:
                    time.sleep(0.05)
                    continue

            npy_path = self._npy_files[self._npy_idx]
            self._npy_idx += 1

            t0 = time.monotonic()
            arr = np.load(str(npy_path))
            image = self._to_hwc(arr)
            preview = self._to_preview_bgr(image)

            self._frame_counter += 1
            packet = FramePacket(
                frame_id=self._frame_counter,
                timestamp=time.monotonic(),
                image=image,
                camera_meta={
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "channels": int(image.shape[2]) if image.ndim == 3 else 1,
                    "source_path": str(npy_path),
                    "preview_image": preview,
                },
            )
            with self._lock:
                self._latest_frame = packet

            sleep_left = period - (time.monotonic() - t0)
            if sleep_left > 0:
                time.sleep(sleep_left)

    def _adapt_camera_frame(self, frame: np.ndarray):
        """Return model input frame and BGR preview based on input mode."""
        if self._input_mode == "hsi_camera":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            r = rgb[:, :, 0]
            g = rgb[:, :, 1]
            b = rgb[:, :, 2]
            intensity = (r + g + b) / 3.0
            rg = r - g
            bg = b - g
            hsi_like = np.stack([r, g, b, intensity, rg, bg], axis=2).astype(np.float32)
            return hsi_like, frame
        return frame, frame

    @staticmethod
    def _to_hwc(arr: np.ndarray) -> np.ndarray:
        """Normalize loaded npy tensor to HWC float32."""
        x = np.asarray(arr)
        if x.ndim == 2:
            x = x[:, :, np.newaxis]
        elif x.ndim == 3 and x.shape[0] <= 16 and x.shape[2] > 16:
            x = np.transpose(x, (1, 2, 0))
        return x.astype(np.float32, copy=False)

    @staticmethod
    def _to_preview_bgr(image: np.ndarray) -> np.ndarray:
        """Build a displayable BGR preview from multi-channel tensors."""
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        C = image.shape[2]
        if C == 1:
            chan = CaptureSource._norm01(image[:, :, 0])
            rgb = np.stack([chan, chan, chan], axis=2)
        elif C >= 3:
            rgb = np.stack([
                CaptureSource._norm01(image[:, :, 0]),
                CaptureSource._norm01(image[:, :, 1]),
                CaptureSource._norm01(image[:, :, 2]),
            ], axis=2)
        else:
            a = CaptureSource._norm01(image[:, :, 0])
            b = CaptureSource._norm01(image[:, :, 1])
            rgb = np.stack([a, b, a], axis=2)
        return (rgb[:, :, ::-1] * 255.0).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _norm01(x: np.ndarray) -> np.ndarray:
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi - lo < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo)).astype(np.float32)
